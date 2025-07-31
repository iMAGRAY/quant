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
- `model.onnx` — FP32
- `model.int8.onnx` — квантованный
- `model.opt.onnx` — дополнительно оптимизированный граф

## Инференс
```python
import onnxruntime as ort, numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
text = "Hello world"
inputs = tokenizer(text, return_tensors="np")
sess = ort.InferenceSession("onnx/qwen3/model.opt.onnx", providers=["CPUExecutionProvider"])
emb = sess.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})[0]
print(emb.shape)
```