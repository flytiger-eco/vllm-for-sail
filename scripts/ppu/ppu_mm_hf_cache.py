#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manage the offline Hugging Face cache used by multimodal tests.

``link`` exposes NAS checkpoints and CI-staged metadata through the HF cache
layout. Hash-shaped snapshot names are required by transformers remote code.
``check`` reports unavailable requirements without downloading anything.
``prefetch`` can populate a cache from a networked host; processing tests only
need configuration/tokenizer/processor files, while generation needs weights.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pathlib
import string
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# mode: full=需要权重, config=load_format=dummy 只需配置。
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


def cmd_link(args: argparse.Namespace) -> int:
    cache = _hub_cache(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    mapping = _model_map()
    for hf_id in GENERATION_MODELS:
        if hf_id in mapping or os.path.exists(
                os.path.join(_repo_dir(cache, hf_id), "refs", "main")):
            continue
        found = _probe_nas(hf_id)
        if found:
            mapping[hf_id] = found

    created = skipped = missing = 0
    for hf_id, local_path in mapping.items():
        if not os.path.isdir(local_path):
            print(f"[link] MISS  {hf_id} -> {local_path} (tests using it will fail)")
            missing += 1
            continue
        repo_dir = _repo_dir(cache, hf_id)
        refs_file = os.path.join(repo_dir, "refs", "main")
        legacy = os.path.join(repo_dir, "snapshots", "main")
        # 只迁移本脚本旧版创建的 snapshots/main，不覆盖真实 HF cache。
        ours = False
        if (os.path.islink(legacy)
                and os.path.realpath(legacy) == os.path.realpath(local_path)):
            try:
                ours = pathlib.Path(refs_file).read_text().strip() == "main"
            except FileNotFoundError:
                ours = True
        if os.path.exists(refs_file) and not ours:
            skipped += 1
            continue
        # trust_remote_code 的动态模块要求 snapshot 名是 40 位 hash。
        fake_rev = hashlib.sha1(local_path.encode()).hexdigest()
        snap_link = os.path.join(repo_dir, "snapshots", fake_rev)
        os.makedirs(os.path.dirname(snap_link), exist_ok=True)
        os.makedirs(os.path.dirname(refs_file), exist_ok=True)
        if not os.path.lexists(snap_link):
            os.symlink(local_path, snap_link)
        elif os.path.realpath(snap_link) != os.path.realpath(local_path):
            raise RuntimeError(
                f"snapshot points elsewhere: {snap_link} -> {os.path.realpath(snap_link)}")
        with open(refs_file, "w") as rf:
            rf.write(fake_rev)
        created += 1
        print(f"[link] OK    {hf_id} -> {local_path} (rev {fake_rev[:12]})")
    print(f"[link] symlinks: created={created} skipped={skipped} "
          f"missing={missing} (total {len(mapping)})")
    return 0


def _requirements() -> list[tuple[str, str, str, str]]:
    """返回 (kind, repo_id, revision, mode)，mode 为 config/full/数据集文件名。"""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.models.multimodal.processing.test_common import get_model_ids_to_test
    from tests.models.registry import HF_EXAMPLE_MODELS

    reqs: dict[tuple[str, str, str], str] = {}

    def _add(kind: str, repo_id: str, rev: str | None, mode: str) -> None:
        key = (kind, repo_id, rev or "-")
        if reqs.get(key) != "full":
            reqs[key] = mode

    for model_id in get_model_ids_to_test():
        info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
        if not info.is_available_online:
            continue
        _add("model", model_id, info.revision, "config")
        if info.tokenizer and info.tokenizer != model_id:
            _add("model", info.tokenizer, None, "config")
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
    # 对只下载配置的 snapshot，逐文件解析，避免 IncompleteSnapshotError 误报。
    snap = None
    for fname in CONFIG_FILES:
        path = try_to_load_from_cache(repo_id, fname, cache_dir=cache,
                                      revision=revision)
        if isinstance(path, str):
            snap = os.path.dirname(path)
            break
    if snap is None:
        if os.path.isdir(_repo_dir(cache, repo_id)):
            return "no-config (repo dir present, revision/config unresolved)"
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


def cmd_prefetch(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = _hub_cache(args.cache_dir)
    rows = _read_list(args.list) if args.list else _requirements()
    failed = 0
    for kind, repo_id, rev, mode in rows:
        revision = None if rev == "-" else rev
        try:
            if kind == "dataset":
                hf_hub_download(repo_id, mode, repo_type="dataset",
                                revision=revision, cache_dir=cache)
            else:
                snapshot_download(
                    repo_id, revision=revision, cache_dir=cache,
                    ignore_patterns=WEIGHT_PATTERNS if mode == "config"
                    else ["original/*"])
            print(f"[prefetch] OK    {kind} {repo_id}@{rev} ({mode})")
        except Exception as e:
            failed += 1
            print(f"[prefetch] FAIL  {kind} {repo_id}@{rev} ({mode}): "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    print(f"[prefetch] {len(rows) - failed}/{len(rows)} fetched into {cache}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_link = sub.add_parser("link")
    p_check = sub.add_parser("check")
    p_check.add_argument("--out")
    p_check.add_argument("--strict", action="store_true",
                         help="exit 1 when anything is missing")
    p_pre = sub.add_parser("prefetch")
    p_pre.add_argument("--list", help="TSV written by `check --out`; "
                       "default: recompute the full requirement set")
    for p in (p_link, p_check, p_pre):
        p.add_argument("--cache-dir", help="default: $HF_HUB_CACHE")
    args = parser.parse_args()
    return {"link": cmd_link, "check": cmd_check,
            "prefetch": cmd_prefetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
