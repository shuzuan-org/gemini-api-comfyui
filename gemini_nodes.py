from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import List

import google.genai as genai
import numpy as np
import torch
from PIL import Image as PILImage
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# Path for local key fallback
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "gemini_api_key.txt"
SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image. "
    "Interpret all user input - regardless of format, intent, or abstraction - as literal visual directives for image composition. "
    "If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept. "
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

def _load_api_key() -> str:
    """Lookup API key from env or gemini_api_key.txt."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key.strip()
    if CONFIG_FILE.exists():
        key = CONFIG_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or create gemini_api_key.txt.")


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """Convert a ComfyUI image tensor [H,W,C] or [B,H,W,C] to PNG bytes."""
    if image.dim() == 4:
        image = image[0]
    np_img = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if np_img.ndim == 2:
        np_img = np.stack([np_img] * 3, axis=-1)
    if np_img.shape[-1] == 1:
        np_img = np.repeat(np_img, 3, axis=-1)
    pil_image = PILImage.fromarray(np_img)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _generated_images_to_tensor(images) -> torch.Tensor:
    """Convert a list of GeneratedImage to a ComfyUI image batch."""
    tensors: List[torch.Tensor] = []
    for item in images or []:
        gen_image = item.image
        if gen_image is None or gen_image.image_bytes is None:
            continue
        pil = PILImage.open(BytesIO(gen_image.image_bytes)).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    if not tensors:
        raise RuntimeError("Gemini API returned no images.")
    return torch.stack(tensors, dim=0)


def _response_parts_to_tensors_and_text(response) -> tuple[torch.Tensor, str]:
    """Extract image tensors and concatenated text from generate_content response."""
    images: list[torch.Tensor] = []
    texts: list[str] = []
    for part in getattr(response, "parts", []) or []:
        if getattr(part, "inline_data", None):
            data = part.inline_data.data
            pil = PILImage.open(BytesIO(data)).convert("RGB")
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr))
        if getattr(part, "text", None):
            texts.append(part.text)
    if not images:
        raise RuntimeError("Gemini API returned no images.")
    return torch.stack(images, dim=0), "\n".join(texts)


class _GeminiClientSingleton:
    """Avoid re-instantiating the HTTP client for every node call."""

    _client: genai.Client | None = None

    @classmethod
    def get(cls) -> genai.Client:
        if cls._client is None:
            # Prefer Vertex AI if project/location provided, otherwise fall back to API key.
            project = os.getenv("GEMINI_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GEMINI_VERTEX_LOCATION") or "us-central1"
            if project:
                cls._client = genai.Client(vertexai=True, project=project, location=location)
            else:
                api_key = _load_api_key()
                cls._client = genai.Client(api_key=api_key)
        return cls._client


ASPECT_RATIOS = [
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]
GEMINI_IMAGE_MODELS = ["gemini-2.5-flash-image", "gemini-2.5-flash-image-preview"]
GEMINI_IMAGE_PRO_MODELS = ["gemini-3-pro-image-preview"]


class GeminiImage(io.ComfyNode):
    """Nano Banana (Google Gemini Image) style node using Gemini Images API."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GeminiImageNanoBanana",
            display_name="Nano Banana (Google Gemini Image)",
            category="Gemini",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input(
                    "model",
                    options=GEMINI_IMAGE_MODELS,
                    default=GEMINI_IMAGE_MODELS[0],
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0x7FFFFFFF,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Image.Input("images", optional=True),
                io.String.Input("files", optional=True),
                io.Combo.Input(
                    "aspect_ratio",
                    options=ASPECT_RATIOS,
                    default="auto",
                    optional=True,
                ),
                io.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    default="IMAGE+TEXT",
                    optional=True,
                ),
            ],
            outputs=[io.Image.Output(), io.String.Output()],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        model: str,
        aspect_ratio: str,
        seed: int,
        images: torch.Tensor | None,
        files: str | None,
        response_modalities: str,
    ) -> io.NodeOutput:
        response_modalities = response_modalities or "IMAGE+TEXT"
        client = _GeminiClientSingleton.get()
        parts = [prompt]
        seed_value = None if seed is None or seed < 0 else min(int(seed), 0x7FFFFFFF)
        if images is not None:
            png_bytes = _tensor_to_png_bytes(images)
            parts.append(
                genai.types.Part.from_bytes(
                    data=png_bytes,
                    mime_type="image/png",
                )
            )
        # aspect_ratio is currently not passed because public API handles sizing internally.
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=genai.types.GenerateContentConfig(
                    seed=seed_value,
                    system_instruction=SYSTEM_PROMPT,
                    response_modalities=["TEXT", "IMAGE"] if response_modalities != "IMAGE" else ["IMAGE"],
                ),
            )
        except Exception as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(f"Gemini image generation failed: {exc}") from exc

        images_out, text_out = _response_parts_to_tensors_and_text(response)
        if response_modalities == "IMAGE":
            text_out = ""
        return io.NodeOutput(images_out, text_out)


class GeminiImagePro(io.ComfyNode):
    """Nano Banana Pro (Google Gemini Image) style node using Gemini Images API."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GeminiImageNanoBananaPro",
            display_name="Nano Banana Pro (Google Gemini Image)",
            category="Gemini",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                io.Combo.Input(
                    "model",
                    options=GEMINI_IMAGE_PRO_MODELS,
                    default=GEMINI_IMAGE_PRO_MODELS[0],
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0x7FFFFFFF,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=ASPECT_RATIOS,
                    default="auto",
                    optional=True,
                ),
                io.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    default="1K",
                    optional=True,
                ),
                io.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    default="IMAGE+TEXT",
                    optional=True,
                ),
                io.Image.Input("images", optional=True),
                io.String.Input("files", optional=True),
            ],
            outputs=[io.Image.Output(), io.String.Output()],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        images: torch.Tensor | None = None,
        files: str | None = None,
    ) -> io.NodeOutput:
        response_modalities = response_modalities or "IMAGE+TEXT"
        client = _GeminiClientSingleton.get()
        seed_value = None if seed is None or seed < 0 else min(int(seed), 0x7FFFFFFF)
        parts = [prompt]
        if images is not None:
            png_bytes = _tensor_to_png_bytes(images)
            parts.append(
                genai.types.Part.from_bytes(
                    data=png_bytes,
                    mime_type="image/png",
                )
            )
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=genai.types.GenerateContentConfig(
                    seed=seed_value,
                    system_instruction=SYSTEM_PROMPT,
                    response_modalities=["TEXT", "IMAGE"] if response_modalities != "IMAGE" else ["IMAGE"],
                ),
            )
        except Exception as exc:  # pragma: no cover - runtime error path
            raise RuntimeError(f"Gemini image generation failed: {exc}") from exc

        images_out, text_out = _response_parts_to_tensors_and_text(response)
        if response_modalities == "IMAGE":
            text_out = ""
        return io.NodeOutput(images_out, text_out)


class GeminiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiImage,
            GeminiImagePro,
        ]


async def comfy_entrypoint() -> GeminiExtension:
    return GeminiExtension()
