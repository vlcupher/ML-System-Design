"""Abstract interface for VLM model adapters."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class ModelAdapter(ABC):
    """Adapter interface: load model, run VLM (image+text) and text-only inference."""

    @abstractmethod
    def load(self) -> None:
        """Load model and processor (called once at startup or lazily)."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is ready for inference."""
        pass

    @abstractmethod
    def run_vlm(self, image: Image.Image, prompt: str, max_new_tokens: int = 650) -> str:
        """Run vision-language inference. Returns generated text."""
        pass

    @abstractmethod
    def run_text_only(self, prompt: str, max_new_tokens: int = 850) -> str:
        """Run text-only generation (e.g. for JSON repair). Returns generated text."""
        pass
