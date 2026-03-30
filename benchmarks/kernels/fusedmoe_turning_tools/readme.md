```bash
export CUDA_VISIBLE_DEVICES=7
ppu-smi -lpc 1300 -i ${CUDA_VISIBLE_DEVICES}

# run tuning for some batch size
python3 faster_autotuning_for_vllm_fusedmoe.py --model Qwen3.5-plus --tp-size 4 --input-metadata ./input_metadata.json --batch-size 128 4094 --dtype bfloat16 --quant-config int8_w8a8

# run tuning for all batch size in input_metadata.json
python3 faster_autotuning_for_vllm_fusedmoe.py --model Qwen3.5-plus --tp-size 4 --input-metadata ./input_metadata.json --dtype bfloat16 --quant-config fp8_w8a8

```