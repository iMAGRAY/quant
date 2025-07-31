#!/usr/bin/env python
"""convert_qwen3_reranker_to_onnx.py

Экспортирует модель `Qwen/Qwen3-Reranker-0.6B` в формат ONNX и выполняет
динамическое квантование весов до INT8, что позволяет значительно снизить
объём файла практически без потери качества.

Пример использования:

    python convert_qwen3_reranker_to_onnx.py \
        --model-id Qwen/Qwen3-Reranker-0.6B \
        --output-dir onnx/qwen3_reranker

Для моделей >2 ГиБ добавьте флаг `--external-data-format`, чтобы ONNX был
сохранён во «внешнем» формате данных.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transformers import AutoTokenizer

try:
    # Опциональная зависимость: optimum + onnxruntime
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification,
        ORTQuantizer,
        QuantizationConfig,
    )
except ImportError as err:  # pragma: no cover
    raise SystemExit(
        "❌  Библиотека `optimum` не найдена. Установите командой:\n"
        "    pip install --upgrade \"optimum[onnxruntime,export]\""
    ) from err

logging.basicConfig(
    format="[%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stdout
)
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_OUTPUT_DIR = "onnx/qwen3_reranker"


def export_and_quantize(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    use_external_data_format: bool = False,
    dynamic: bool = True,
) -> None:
    """Экспорт и квантование модели.

    Parameters
    ----------
    model_id
        HuggingFace-идентификатор модели или путь к локальной папке.
    output_dir
        Папка назначения для артефактов ONNX.
    use_external_data_format
        Сохранять ли модель ONNX во «внешнем» формате данных (>2 ГиБ).
    dynamic
        Выполнять ли динамическое INT8-квантование (по умолчанию ‑ да).
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("🚀 Экспортируем %s → ONNX…", model_id)

    # Параметр `use_external_data_format` появился в недавних версиях `optimum`.
    # Чтобы оставаться совместимыми с более старыми версиями, проверяем
    # поддерживается ли он рефлексивно. Если нет — вызываем без параметра.
    import inspect  # локальный импорт, чтобы не засорять глобальное пространство

    _fp_sig = inspect.signature(ORTModelForSequenceClassification.from_pretrained)
    kwargs: dict[str, object] = {
        "model_id_or_path": model_id,  # позиционный будет передан как *args
        "export": True,
    }
    if "use_external_data_format" in _fp_sig.parameters:
        kwargs["use_external_data_format"] = use_external_data_format
    else:
        if use_external_data_format:
            LOGGER.warning(
                "Версия optimum не поддерживает `use_external_data_format`; "
                "ONNX будет сохранён в обычном формате. Обновите пакет: "
                "pip install -U 'optimum[onnxruntime,export]'"
            )

    # Разворачиваем kwargs аккуратно (без лишних параметров)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        kwargs.pop("model_id_or_path"),
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("✅ ONNX-модель сохранена в %s", output_dir)

    if dynamic:
        LOGGER.info("⚙️  Динамическое INT8-квантование…")
        try:
            qconfig = QuantizationConfig.for_dynamic()  # type: ignore[attr-defined]
            quantizer = ORTQuantizer.from_pretrained(ort_model)
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig,
            )
            LOGGER.info("✅ Квантованная модель сохранена в %s", output_dir)
        except AttributeError:
            # Старые версии optimum (<1.18) не имеют метода for_dynamic.
            LOGGER.warning(
                "QuantizationConfig.for_dynamic не найден – переключаемся на "
                "onnxruntime.quantization.quantize_dynamic. Рекомендуется "
                "обновить optimum до последней версии."
            )

            from onnxruntime.quantization import QuantType, quantize_dynamic

            onnx_model_path = next(output_dir.glob("*.onnx"))
            quant_model_path = output_dir / "model.int8.onnx"

            quantize_dynamic(
                model_input=onnx_model_path.as_posix(),
                model_output=quant_model_path.as_posix(),
                weight_type=QuantType.QInt8,
                per_channel=False,
                op_types_to_quantize=["MatMul", "Gemm"],
                reduce_range=True,
            )
            LOGGER.info("✅ Квантованная модель сохранена в %s", quant_model_path)

    LOGGER.info("🎉 Готово!")


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(
            "Экспорт Qwen3-Reranker-0.6B в ONNX c динамическим INT8-квантованием"
        )
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HF-идентификатор или путь (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Папка назначения (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--external-data-format",
        action="store_true",
        help="Сохранять ONNX во внешнем формате данных (>2 ГиБ)",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Отключить динамическое квантование",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    export_and_quantize(
        model_id=args.model_id,
        output_dir=args.output_dir,
        use_external_data_format=args.external_data_format,
        dynamic=not args.no_dynamic,
    )


if __name__ == "__main__":  # pragma: no cover
    main()