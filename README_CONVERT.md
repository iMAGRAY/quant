# Qwen3 Embedding 0.6B → ONNX INT8

## Быстрый старт
```bash
# Установка зависимостей (python>=3.10)
pip install -r requirements.txt

# Конвертация (пример CPU)
python convert_qwen3_embedding_to_onnx.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --out_dir ./onnx/qwen3 \
  --device cpu
```

Артефакты:
* `model.onnx` — FP32
* `model.int8.onnx` — квантованный
* `model.opt.onnx` — дополнительно оптимизированный граф

## Инференс
```