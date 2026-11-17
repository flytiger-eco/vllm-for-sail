#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manage the offline Hugging Face cache used by multimodal tests.

``link`` exposes NAS checkpoints and CI-staged metadata through the HF cache
layout. Hash-shaped snapshot names are required by transformers remote code.
``requirements`` exports the release's processing requirements without vLLM.
``check`` reports unavailable requirements without downloading anything.
``prefetch`` can populate a cache or artifact directory from a networked host;
processing tests only need configuration/tokenizer/processor files, while
generation needs weights.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pathlib
import string
import sys
import time
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# mode: full=需要权重, config=模型配置, tokenizer=独立分词器仓库。
GENERATION_MODELS = {
    "google/gemma-3-4b-it": "full",
    "OpenMOSS-Team/MOSS-Audio-4B-Instruct": "full",
    "Qwen/Qwen2-VL-2B-Instruct": "full",
    "Qwen/Qwen2.5-VL-3B-Instruct": "full",
    "Qwen/Qwen3-VL-2B-Instruct": "full",
    "Qwen/Qwen3.5-0.8B": "full",
    "OpenGVLab/InternVL3-1B": "full",
    "zai-org/GLM-4.1V-9B-Thinking": "config",
}

VIDEO_DATASET = ("raushan-testing-hf/videos-test", "sample_demo_1.mp4")

# 可用 PPU_MM_MODEL_MAP="org/a=/path/a,org/b=/path/b" 追加或覆盖。
MODEL_MAP = {
    "nvidia/NVIDIA-Nemotron-Parse-v1.1":
        "/nas_aisw/datasets/checkpoints/LLM/NVIDIA/v1.1/NVIDIA-Nemotron-Parse-v1.1",
    "royokong/e5-v": "/nas_aisw/datasets/checkpoints/LLM/e5/v1.0/e5-v",
}
NAS_CKPT_ROOT = os.environ.get(
    "PPU_MM_CKPT_ROOT", "/nas_aisw/datasets/checkpoints/LLM")

WEIGHT_PATTERNS = [
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx",
    "*.h5", "*.msgpack", "*.pdparams", "*.tflite", "original/*",
]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
CONFIG_FILES = ("config.json", "params.json")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer.model", "spiece.model",
    "sentencepiece.bpe.model", "vocab.json", "vocab.txt",
)
METADATA_PATTERNS = [
    "*.json", "*.model", "tokenizer.model*", "*.txt", "*.jinja", "*.jinja2",
    "*.py", "*.tiktoken", "*.merges", "*.vocab", "*.yaml", "*.yml",
    "*.chat_template", "*.spm", "*.sentencepiece",
]


def _hub_cache(arg: str | None) -> str:
    return arg or os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub")


def _repo_dir(cache: str, repo_id: str, kind: str = "model") -> str:
    return os.path.join(cache, f"{kind}s--" + repo_id.replace("/", "--"))


def _is_commit_hash(name: str) -> bool:
    return len(name) == 40 and all(c in string.hexdigits for c in name)


def _model_map() -> dict[str, str]:
    mapping = dict(MODEL_MAP)
    for item in filter(None, os.environ.get("PPU_MM_MODEL_MAP", "").split(",")):
        hf_id, _, path = item.partition("=")
        if hf_id.strip() and path.strip():
            mapping[hf_id.strip()] = path.strip()
    return mapping


def _probe_nas(hf_id: str) -> str | None:
    """在 <root>/<vendor>[/<ver>]/<name> 下查找本地 checkpoint。"""
    org, name = hf_id.split("/", 1)
    cands = sorted(
        p for pat in (f"{NAS_CKPT_ROOT}/*/*/{name}", f"{NAS_CKPT_ROOT}/*/{name}")
        for p in glob.glob(pat)
        if any(os.path.isfile(os.path.join(p, f)) for f in CONFIG_FILES))
    if not cands:
        return None
    cands.sort(key=lambda p: org.lower() not in p.lower())
    if len(cands) > 1:
        print(f"[link] {hf_id}: {len(cands)} candidates, using {cands[0]}")
    return cands[0]


def _staged_models() -> dict[str, str]:
    root = REPO_ROOT / "ci-wheel" / "hf-models"
    models = {}
    if not root.is_dir():
        return models
    for org_dir in sorted(root.iterdir()):
        if not org_dir.is_dir():
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if model_dir.is_dir() and any(
                    (model_dir / name).is_file()
                    for name in CONFIG_FILES + TOKENIZER_FILES):
                models[f"{org_dir.name}/{model_dir.name}"] = str(model_dir)
    return models


def cmd_link(args: argparse.Namespace) -> int:
    cache = _hub_cache(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    revisions = {}
    for kind, hf_id, rev, _ in _requirements():
        if kind == "model":
            revisions.setdefault(hf_id, set()).add(rev)
    mapping = _staged_models()
    # Processing metadata must not shadow a generation checkpoint with weights.
    # Explicit MODEL_MAP entries remain the final override for local debugging.
    for hf_id in GENERATION_MODELS:
        found = _probe_nas(hf_id)
        if found:
            mapping[hf_id] = found
    mapping.update(_model_map())

    created = skipped = missing = 0
    for hf_id, local_path in mapping.items():
        if not os.path.isdir(local_path):
            print(f"[link] MISS  {hf_id} -> {local_path} (tests using it will fail)")
            missing += 1
            continue
        repo_dir = _repo_dir(cache, hf_id)
        # Keep main plus every revision pinned by the release's model registry.
        for rev in sorted(revisions.get(hf_id, set()) | {"-"}):
            ref_name = "main" if rev == "-" else rev
            refs_file = os.path.join(repo_dir, "refs", ref_name)
            try:
                existing_rev = pathlib.Path(refs_file).read_text().strip()
            except FileNotFoundError:
                existing_rev = ""
            existing_snap = os.path.join(repo_dir, "snapshots", existing_rev)
            if (_is_commit_hash(existing_rev) and os.path.isdir(existing_snap)
                    and any(os.path.isfile(os.path.join(existing_snap, name))
                            for name in CONFIG_FILES + TOKENIZER_FILES)):
                skipped += 1
                continue
            # trust_remote_code requires a 40-hex snapshot, not snapshots/main.
            fake_rev = hashlib.sha1(local_path.encode()).hexdigest()
            snap_link = os.path.join(repo_dir, "snapshots", fake_rev)
            os.makedirs(os.path.dirname(snap_link), exist_ok=True)
            os.makedirs(os.path.dirname(refs_file), exist_ok=True)
            if not os.path.lexists(snap_link):
                os.symlink(local_path, snap_link)
            elif os.path.realpath(snap_link) != os.path.realpath(local_path):
                raise RuntimeError(
                    f"snapshot points elsewhere: {snap_link} -> "
                    f"{os.path.realpath(snap_link)}")
            pathlib.Path(refs_file).write_text(fake_rev)
            created += 1
            print(f"[link] OK    {hf_id}@{ref_name} -> {local_path} "
                  f"(rev {fake_rev[:12]})")
    print(f"[link] refs: created={created} skipped={skipped} "
          f"missing={missing} (models {len(mapping)})")
    return 0


def _load_registry(lightweight: bool):
    if lightweight:
        # The build runner only needs the registry dataclasses. Avoid importing
        # the compiled vLLM wheel and its device runtime just for type aliases.
        for name in ("vllm", "vllm.config", "vllm.config.model"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["vllm.config.model"].ModelDType = str
        sys.modules["vllm.config.model"].TokenizerMode = str
    from tests.models import registry
    return registry


def _processing_model_ids(registry):
    """Mirror processing/test_common.py without importing device modules."""
    transformers_arch_ids = {
        model_id
        for info in registry._TRANSFORMERS_BACKEND_MODELS.values()
        for model_id in (info.default, *info.extras.values())
    }
    vllm_only_archs = {
        arch
        for arch, info in registry._MULTIMODAL_EXAMPLE_MODELS.items()
        if not any(
            model_id in transformers_arch_ids
            for model_id in (info.default, *info.extras.values())
        )
    }
    for arch in vllm_only_archs:
        info = registry.HF_EXAMPLE_MODELS.get_hf_info(arch)
        yield info.default
        for extra_type, model_id in info.extras.items():
            if "fp" not in extra_type:
                yield model_id


def _requirements(lightweight: bool = False,
                  include_generation: bool = True
                  ) -> list[tuple[str, str, str, str]]:
    """返回 (kind, repo_id, revision, mode)，mode 为 config/full/数据集文件名。"""
    sys.path.insert(0, str(REPO_ROOT))
    registry = _load_registry(lightweight)
    reqs: dict[tuple[str, str, str], str] = {}

    def _add(kind: str, repo_id: str, rev: str | None, mode: str) -> None:
        key = (kind, repo_id, rev or "-")
        priority = {"tokenizer": 0, "config": 1, "full": 2}
        if (key not in reqs
                or priority.get(mode, 0) > priority.get(reqs[key], 0)):
            reqs[key] = mode

    for model_id in _processing_model_ids(registry):
        info = registry.HF_EXAMPLE_MODELS.find_hf_info(model_id)
        if not info.is_available_online:
            continue
        _add("model", model_id, info.revision, "config")
        if info.tokenizer and info.tokenizer != model_id:
            _add("model", info.tokenizer, None, "tokenizer")
    if include_generation:
        for model_id, mode in GENERATION_MODELS.items():
            _add("model", model_id, None, mode)
    _add("dataset", VIDEO_DATASET[0], None, VIDEO_DATASET[1])
    return [(k, r, rev, m) for (k, r, rev), m in sorted(reqs.items())]


def _read_list(path: str) -> list[tuple[str, str, str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                kind, repo_id, rev, mode = line.rstrip("\n").split("\t")[:4]
                rows.append((kind, repo_id, rev, mode))
    return rows


def cmd_requirements(args: argparse.Namespace) -> int:
    rows = _requirements(
        lightweight=args.lightweight,
        include_generation=not args.processing_only,
    )
    text = "# kind\trepo_id\trevision\tmode\n" + "".join(
        "\t".join(row) + "\n" for row in rows)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pathlib.Path(args.out).write_text(text)
        print(f"[requirements] {len(rows)} rows -> {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _check_one(cache: str, kind: str, repo_id: str, rev: str,
               mode: str) -> str | None:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    revision = None if rev == "-" else rev
    if kind == "dataset":
        try:
            hf_hub_download(repo_id, mode, repo_type="dataset", revision=revision,
                            cache_dir=cache, local_files_only=True)
            return None
        except Exception as e:  # LocalEntryNotFoundError
            return f"absent ({type(e).__name__})"
    # 对只下载元数据的 snapshot，逐文件解析，避免 IncompleteSnapshotError 误报。
    snap = None
    required_files = TOKENIZER_FILES if mode == "tokenizer" else CONFIG_FILES
    for fname in required_files:
        path = try_to_load_from_cache(repo_id, fname, cache_dir=cache,
                                      revision=revision)
        if isinstance(path, str):
            snap = os.path.dirname(path)
            break
    if snap is None:
        if os.path.isdir(_repo_dir(cache, repo_id)):
            kind_missing = "tokenizer" if mode == "tokenizer" else "config"
            return f"no-{kind_missing} (repo dir present, revision/files unresolved)"
        return "absent"
    if not _is_commit_hash(os.path.basename(snap)):
        return "non-hash-snapshot (breaks trust_remote_code)"
    if mode == "full" and not any(
            f.endswith(WEIGHT_SUFFIXES) for f in os.listdir(snap)):
        return "no-weights"
    return None


def cmd_check(args: argparse.Namespace) -> int:
    cache = _hub_cache(args.cache_dir)
    reqs = _requirements()
    bad = []
    for kind, repo_id, rev, mode in reqs:
        reason = _check_one(cache, kind, repo_id, rev, mode)
        if reason:
            bad.append((kind, repo_id, rev, mode, reason))
            print(f"[check] MISS  {kind} {repo_id}@{rev} ({mode}): {reason}")
    print(f"[check] {len(reqs) - len(bad)}/{len(reqs)} available offline "
          f"in {cache}; {len(bad)} missing")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("# kind\trepo_id\trevision\tmode\treason\n")
            for row in bad:
                f.write("\t".join(row) + "\n")
        print(f"[check] missing list -> {args.out} (feed it to `prefetch --list`)")
    return 1 if (args.strict and bad) else 0


def _local_prefetch_complete(local_dir: str, kind: str, mode: str) -> bool:
    if kind == "dataset":
        return os.path.isfile(os.path.join(local_dir, mode))
    required_files = TOKENIZER_FILES if mode == "tokenizer" else CONFIG_FILES
    return any(os.path.isfile(os.path.join(local_dir, name))
               for name in required_files)


def _exception_chain(error: Exception) -> list[Exception]:
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _retryable_prefetch_error(error: Exception) -> bool:
    for cause in _exception_chain(error):
        response = getattr(cause, "response", None)
        status = getattr(response, "status_code", None)
        if (status == 429 or (status is not None and status >= 500)
                or "429 Too Many Requests" in str(cause)):
            return True
    return False


def _format_prefetch_error(error: Exception) -> str:
    chain = _exception_chain(error)
    selected = chain[:1] if len(chain) == 1 else [chain[0], chain[-1]]
    parts = []
    for cause in selected:
        lines = str(cause).splitlines() or [type(cause).__name__]
        detail = lines[0][:200]
        if len(lines) > 1:
            detail += " | " + lines[-1][:200]
        parts.append(f"{type(cause).__name__}: {detail}")
    return " | caused by ".join(parts)


def _gated_prefetch_error(error: Exception) -> bool:
    """True when the failure is an authorization gate (401/403).

    Gated HF repos return 401/403 without a PPU_HF_TOKEN. That is an expected,
    tolerable gap for a build-only metadata prefetch and must not fail the whole
    step, which would upload no wheel artifact and skip every area's tests.
    Real transport/missing-file errors stay fatal.
    """
    for cause in _exception_chain(error):
        response = getattr(cause, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403):
            return True
        text = str(cause)
        if type(cause).__name__ == "GatedRepoError" or (
            "is restricted" in text
            or "401 Unauthorized" in text
            or "403 Forbidden" in text
        ):
            return True
    return False


def cmd_prefetch(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    if args.retries < 1 or args.max_workers < 1 or args.retry_delay < 0:
        raise ValueError("retries/max-workers must be positive and retry-delay >= 0")
    cache = _hub_cache(args.cache_dir)
    rows = _read_list(args.list) if args.list else _requirements()
    failed = 0
    gated = 0
    for kind, repo_id, rev, mode in rows:
        revision = None if rev == "-" else rev
        local_dir = None
        marker = None
        if args.local_root:
            root = args.local_root
            if kind == "dataset":
                root = args.dataset_root or os.path.join(root, "_datasets")
            local_dir = os.path.join(root, *repo_id.split("/"))
            marker = os.path.join(local_dir, ".ppu-mm-prefetch-complete")
            try:
                marker_value = pathlib.Path(marker).read_text()
            except FileNotFoundError:
                marker_value = ""
            if (marker_value == f"{rev}\t{mode}"
                    and _local_prefetch_complete(local_dir, kind, mode)):
                print(f"[prefetch] OK    {kind} {repo_id}@{rev} ({mode}) [cached]")
                continue
        for attempt in range(args.retries):
            try:
                if kind == "dataset":
                    hf_hub_download(repo_id, mode, repo_type="dataset",
                                    revision=revision, cache_dir=cache,
                                    local_dir=local_dir)
                else:
                    allow = (METADATA_PATTERNS
                             if args.metadata_only and mode != "full" else None)
                    ignore = (None if allow is not None else
                              WEIGHT_PATTERNS if mode != "full" else ["original/*"])
                    snapshot_download(
                        repo_id, revision=revision, cache_dir=cache,
                        local_dir=local_dir, allow_patterns=allow,
                        ignore_patterns=ignore, max_workers=args.max_workers)
                if (local_dir is not None
                        and not _local_prefetch_complete(local_dir, kind, mode)):
                    raise RuntimeError(
                        f"download returned without required files in {local_dir}")
            except Exception as e:
                if attempt + 1 < args.retries and _retryable_prefetch_error(e):
                    delay = args.retry_delay * (attempt + 1)
                    print(f"[prefetch] RETRY {kind} {repo_id}@{rev} "
                          f"after {delay}s: {type(e).__name__}")
                    time.sleep(delay)
                    continue
                if args.allow_gated and _gated_prefetch_error(e):
                    # Authorization gate (401/403): expected without a token,
                    # recorded but not fatal, so it cannot skip every area.
                    gated += 1
                    print(f"[prefetch] GATED {kind} {repo_id}@{rev} ({mode}): "
                          f"access restricted, skipped without a token")
                    break
                failed += 1
                print(f"[prefetch] FAIL  {kind} {repo_id}@{rev} ({mode}): "
                      f"{_format_prefetch_error(e)}")
                break
            else:
                if marker is not None:
                    pathlib.Path(marker).write_text(f"{rev}\t{mode}")
                print(f"[prefetch] OK    {kind} {repo_id}@{rev} ({mode})")
                break
    destination = args.local_root or cache
    print(f"[prefetch] {len(rows) - failed - gated}/{len(rows)} fetched, "
          f"{gated} gated-skipped, {failed} failed into {destination}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_link = sub.add_parser("link")
    p_check = sub.add_parser("check")
    p_check.add_argument("--out")
    p_check.add_argument("--strict", action="store_true",
                         help="exit 1 when anything is missing")
    p_req = sub.add_parser("requirements")
    p_req.add_argument("--out")
    p_req.add_argument("--lightweight", action="store_true",
                       help="stub vLLM type aliases for a build-only host")
    p_req.add_argument("--processing-only", action="store_true",
                       help="exclude generation checkpoints and their weights")
    p_pre = sub.add_parser("prefetch")
    p_pre.add_argument("--list", help="TSV written by `check --out`; "
                       "default: recompute the full requirement set")
    p_pre.add_argument("--local-root", help="write flat model dirs for an artifact")
    p_pre.add_argument("--dataset-root", help="flat dataset root with --local-root")
    p_pre.add_argument("--metadata-only", action="store_true",
                       help="fetch only processor/tokenizer/config/dynamic code")
    p_pre.add_argument("--allow-gated", action="store_true",
                       help="tolerate 401/403 authorization-gated repos "
                            "(expected without a PPU_HF_TOKEN) instead of failing")
    p_pre.add_argument("--max-workers", type=int, default=4,
                       help="parallel downloads per repository (default: 4)")
    p_pre.add_argument("--retries", type=int, default=3,
                       help="attempts per repository for transient HTTP errors")
    p_pre.add_argument("--retry-delay", type=int, default=10,
                       help="base delay in seconds between HTTP retries")
    for p in (p_link, p_check, p_pre):
        p.add_argument("--cache-dir", help="default: $HF_HUB_CACHE")
    args = parser.parse_args()
    return {"link": cmd_link, "check": cmd_check,
            "requirements": cmd_requirements,
            "prefetch": cmd_prefetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
