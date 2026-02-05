"""Qwen3-VL-2B-Instruct adapter (v4: same pipeline as original, model = Qwen3-VL 2B)."""

import os
from typing import Any

from PIL import Image

from diagram_service.models.base import ModelAdapter

# Full model 4-bit (recommended: complete lm_head) or pre-quantized; override via env
MODEL_ID = os.environ.get("QWEN3VL_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
PROCESSOR_ID = os.environ.get("QWEN3VL_PROCESSOR_ID", "Qwen/Qwen3-VL-2B-Instruct")
DEVICE = os.environ.get("DEVICE", "")  # "cpu" to force CPU
# Use BitsAndBytes 4-bit for full model (avoids missing lm_head in some pre-quantized checkpoints)
USE_4BIT = os.environ.get("QWEN3VL_4BIT", "1").strip().lower() in ("1", "true", "yes")


class Qwen3VLAdapter(ModelAdapter):
    """Adapter for Qwen3-VL-2B-Instruct: load once, run_vlm / run_text_only."""

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None

    def _get_device(self):
        """Device for inputs (model may use device_map)."""
        if self._device is not None:
            return self._device
        if self._model is None:
            return None
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return getattr(self._model, "device", None)

    def load(self) -> None:
        import torch

        use_cuda = torch.cuda.is_available() and DEVICE != "cpu"
        if use_cuda:
            device_map = "cuda:0"  # single GPU: FP4 layers get proper CUDA init
        else:
            device_map = "cpu"

        # On CPU: 4-bit / bnb quantized checkpoints don't work properly → use full model
        load_model_id = MODEL_ID
        if not use_cuda and ("4bit" in MODEL_ID.lower() or "bnb-4bit" in MODEL_ID.lower()):
            load_model_id = PROCESSOR_ID  # full Qwen/Qwen3-VL-2B-Instruct

        self._processor = __import__("transformers").AutoProcessor.from_pretrained(
            PROCESSOR_ID,
        )

        from transformers import Qwen3VLForConditionalGeneration

        if USE_4BIT and use_cuda:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                load_model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
        else:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                load_model_id,
                torch_dtype="auto",
                device_map=device_map,
            )

        self._model.eval()
        if use_cuda:
            self._device = torch.device("cuda", 0)
        else:
            self._device = torch.device("cpu")

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def run_vlm(self, image: Image.Image, prompt: str, max_new_tokens: int = 650) -> str:
        import torch

        if not self.is_loaded():
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self._get_device()
        if device is not None:
            inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        in_ids = getattr(inputs, "input_ids", inputs.get("input_ids") if isinstance(inputs, dict) else None)
        if in_ids is not None and generated_ids.shape[1] > in_ids.shape[1]:
            generated_ids_trimmed = [
                out_ids[in_ids.shape[1] :].tolist()
                for out_ids in generated_ids
            ]
        else:
            generated_ids_trimmed = [generated_ids[i].tolist() for i in range(generated_ids.shape[0])]
        decoded = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else "").strip()

    def run_text_only(self, prompt: str, max_new_tokens: int = 850) -> str:
        import torch

        if not self.is_loaded():
            self.load()

        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self._get_device()
        if device is not None:
            inputs = inputs.to(device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        in_ids = getattr(inputs, "input_ids", inputs.get("input_ids") if isinstance(inputs, dict) else None)
        if in_ids is not None and generated_ids.shape[1] > in_ids.shape[1]:
            generated_ids_trimmed = [
                out_ids[in_ids.shape[1] :].tolist()
                for out_ids in generated_ids
            ]
        else:
            generated_ids_trimmed = [generated_ids[i].tolist() for i in range(generated_ids.shape[0])]
        decoded = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else "").strip()
