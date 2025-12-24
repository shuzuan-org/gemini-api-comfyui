from __future__ import annotations

import os
import math
import logging
import time
import threading
from io import BytesIO
from pathlib import Path
from typing import List
from datetime import datetime, timedelta

import google.genai as genai
import numpy as np
import torch
from PIL import Image as PILImage
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Path for local key fallback
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "gemini_api_key.txt"
SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image. "
    "Interpret all user input - regardless of format, intent, or abstraction - as literal visual directives for image composition. "
    "If a prompt is conversational or lacks specific visual details, you must creatively invent a concrete visual scenario that depicts the concept. "
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

# API调用配置
API_TIMEOUT = int(os.getenv("GEMINI_API_TIMEOUT", "600"))  # 默认600秒超时（10分钟），适配4K高质量图像生成
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))  # 最多重试2次（因为超时时间更长）
RETRY_DELAY = float(os.getenv("GEMINI_RETRY_DELAY", "5.0"))  # 重试延迟5秒

# 断路器配置
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("GEMINI_CB_THRESHOLD", "5"))  # 连续失败5次后断路
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("GEMINI_CB_TIMEOUT", "300"))  # 断路后等待300秒（5分钟）


class CircuitBreaker:
    """断路器模式实现，防止持续的失败请求造成资源耗尽"""

    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD, timeout: int = CIRCUIT_BREAKER_TIMEOUT):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.total_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """通过断路器执行函数调用"""
        with self._lock:
            self.total_count += 1

            # 检查断路器状态
            if self.state == "OPEN":
                if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                    logger.info("断路器从OPEN转换到HALF_OPEN状态，尝试恢复")
                    self.state = "HALF_OPEN"
                else:
                    remaining = self.timeout - (datetime.now() - self.last_failure_time).total_seconds() if self.last_failure_time else 0
                    error_msg = (
                        f"断路器处于OPEN状态，拒绝请求。"
                        f"失败次数: {self.failure_count}/{self.failure_threshold}，"
                        f"总请求数: {self.total_count}，"
                        f"成功率: {self.success_count/self.total_count*100:.1f}%，"
                        f"将在 {remaining:.0f}秒 后重试"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        # 执行函数调用
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.success_count += 1
                self.failure_count = 0  # 重置失败计数
                if self.state == "HALF_OPEN":
                    logger.info("断路器从HALF_OPEN转换到CLOSED状态，恢复正常")
                    self.state = "CLOSED"
            logger.info(f"API调用成功，成功率: {self.success_count/self.total_count*100:.1f}% ({self.success_count}/{self.total_count})")
            return result
        except Exception as exc:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(
                        f"连续失败 {self.failure_count} 次，断路器打开！"
                        f"总请求数: {self.total_count}，"
                        f"成功率: {self.success_count/self.total_count*100:.1f}%。"
                        f"将在 {self.timeout} 秒后尝试恢复"
                    )
                else:
                    logger.warning(
                        f"API调用失败 ({self.failure_count}/{self.failure_threshold})，"
                        f"成功率: {self.success_count/self.total_count*100:.1f}%。"
                        f"错误: {exc}"
                    )
            raise


# 全局断路器实例
_circuit_breaker = CircuitBreaker()

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
    """Convert a ComfyUI image tensor [H,W,C] or [B,H,W,C] to PNG bytes.
    如果是批次输入（4维），只返回第一张图片的PNG bytes（用于向后兼容）。"""
    buffer = None
    pil_image = None
    try:
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
    finally:
        # 确保资源被释放
        if buffer:
            buffer.close()
        if pil_image:
            pil_image.close()


def _tensor_to_png_bytes_list(images: torch.Tensor) -> List[bytes]:
    """Convert a ComfyUI image tensor [H,W,C] or [B,H,W,C] to a list of PNG bytes.
    支持多张图片输入，返回所有图片的PNG bytes列表。"""
    png_bytes_list: List[bytes] = []
    
    # 处理单张图片（3维）或批次图片（4维）
    if images.dim() == 3:
        # 单张图片，添加批次维度
        images = images.unsqueeze(0)
    elif images.dim() != 4:
        raise ValueError(f"不支持的图像张量维度: {images.dim()}，期望3或4维")
    
    # 遍历批次中的每张图片
    batch_size = images.shape[0]
    for i in range(batch_size):
        image = images[i]
        buffer = None
        pil_image = None
        try:
            np_img = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if np_img.ndim == 2:
                np_img = np.stack([np_img] * 3, axis=-1)
            if np_img.shape[-1] == 1:
                np_img = np.repeat(np_img, 3, axis=-1)
            pil_image = PILImage.fromarray(np_img)
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            png_bytes_list.append(buffer.getvalue())
        finally:
            # 确保资源被释放
            if buffer:
                buffer.close()
            if pil_image:
                pil_image.close()
    
    return png_bytes_list


def _generated_images_to_tensor(images) -> torch.Tensor:
    """Convert a list of GeneratedImage to a ComfyUI image batch."""
    tensors: List[torch.Tensor] = []
    for item in images or []:
        gen_image = item.image
        if gen_image is None or gen_image.image_bytes is None:
            continue
        buffer = None
        pil = None
        try:
            buffer = BytesIO(gen_image.image_bytes)
            pil = PILImage.open(buffer).convert("RGB")
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        finally:
            if pil:
                pil.close()
            if buffer:
                buffer.close()
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
            buffer = None
            pil = None
            try:
                buffer = BytesIO(data)
                pil = PILImage.open(buffer).convert("RGB")
                arr = np.asarray(pil, dtype=np.float32) / 255.0
                images.append(torch.from_numpy(arr))
            finally:
                if pil:
                    pil.close()
                if buffer:
                    buffer.close()
        if getattr(part, "text", None):
            texts.append(part.text)
    if not images:
        raise RuntimeError("Gemini API returned no images.")
    return torch.stack(images, dim=0), "\n".join(texts)


def _call_gemini_api_with_retry(client, model: str, contents, config) -> any:
    """
    通过断路器和重试机制调用Gemini API

    Args:
        client: Gemini client实例
        model: 模型名称
        contents: 请求内容
        config: 生成配置

    Returns:
        API响应

    Raises:
        RuntimeError: 当所有重试都失败或断路器打开时
    """
    last_exception = None
    total_attempts = MAX_RETRIES + 1  # 总尝试次数 = 初始1次 + 重试N次

    for attempt in range(total_attempts):
        try:
            is_retry = attempt > 0
            attempt_desc = f"第{attempt}次重试" if is_retry else "初始尝试"
            logger.info(
                f"调用Gemini API ({attempt_desc}，总进度 {attempt + 1}/{total_attempts})，模型: {model}"
            )
            start_time = time.time()

            # 通过断路器调用API
            def _api_call():
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            response = _circuit_breaker.call(_api_call)

            elapsed = time.time() - start_time
            logger.info(f"API调用成功，耗时: {elapsed:.2f}秒")
            return response

        except Exception as exc:
            last_exception = exc
            elapsed = time.time() - start_time

            # 如果是断路器打开的错误，直接抛出，不再重试
            if "断路器处于OPEN状态" in str(exc):
                logger.error(f"断路器已打开，停止重试")
                raise RuntimeError(f"Gemini API调用失败（断路器打开）: {exc}") from exc

            if attempt < total_attempts - 1:
                logger.warning(
                    f"API调用失败 (尝试 {attempt + 1}/{total_attempts})，"
                    f"耗时: {elapsed:.2f}秒，"
                    f"错误: {exc}。"
                    f"{RETRY_DELAY}秒后进行第{attempt + 1}次重试..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error(
                    f"API调用失败，已用尽所有尝试机会（初始1次+重试{MAX_RETRIES}次={total_attempts}次），"
                    f"错误: {exc}"
                )

    # 所有重试都失败
    raise RuntimeError(
        f"Gemini API调用失败，已尝试{total_attempts}次（初始1次+重试{MAX_RETRIES}次）: {last_exception}"
    ) from last_exception


def _aspect_ratio_hint(aspect_ratio: str | None, images: torch.Tensor | None) -> str | None:
    """Return only a supported aspect ratio; fall back to the closest allowed option."""
    allowed = [ratio for ratio in ASPECT_RATIOS if ratio != "auto"]

    def _ratio_to_float(ratio: str) -> float:
        a, b = ratio.split(":")
        return float(a) / float(b)

    if aspect_ratio and aspect_ratio != "auto":
        return aspect_ratio if aspect_ratio in allowed else None
    if images is None:
        return None
    image = images[0] if images.dim() == 4 else images
    if image.dim() != 3:
        return None
    height, width = int(image.shape[0]), int(image.shape[1])
    if height <= 0 or width <= 0:
        return None
    divisor = math.gcd(width, height)
    if divisor == 0:
        return None
    simplified = f"{width // divisor}:{height // divisor}"
    if simplified in allowed:
        return simplified
    # Gemini only accepts a fixed set of aspect ratios. Choose the closest valid one.
    target_ratio = width / height
    closest = min(allowed, key=lambda ratio: abs(_ratio_to_float(ratio) - target_ratio))
    return closest


class _GeminiClientSingleton:
    """Avoid re-instantiating the HTTP client for every node call."""

    _client: genai.Client | None = None

    @classmethod
    def get(cls) -> genai.Client:
        if cls._client is None:
            # 配置HTTP选项，包括超时设置（单位：毫秒）
            http_options = genai.types.HttpOptions(timeout=API_TIMEOUT * 1000)
            
            # Prefer Vertex AI if project/location provided, otherwise fall back to API key.
            project = os.getenv("GEMINI_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GEMINI_VERTEX_LOCATION") or "us-central1"
            if project:
                cls._client = genai.Client(
                    vertexai=True, 
                    project=project, 
                    location=location,
                    http_options=http_options
                )
            else:
                api_key = _load_api_key()
                cls._client = genai.Client(
                    api_key=api_key,
                    http_options=http_options
                )
            logger.info(f"Gemini客户端已初始化，超时设置: {API_TIMEOUT}秒")
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
        ratio_hint = _aspect_ratio_hint(aspect_ratio, images)
        prompt_text = f"{prompt}\nAspect ratio: {ratio_hint}" if ratio_hint else prompt
        parts = [prompt_text]
        seed_value = None if seed is None or seed < 0 else min(int(seed), 0x7FFFFFFF)
        if images is not None:
            # 支持多张图片输入
            png_bytes_list = _tensor_to_png_bytes_list(images)
            for png_bytes in png_bytes_list:
                parts.append(
                    genai.types.Part.from_bytes(
                        data=png_bytes,
                        mime_type="image/png",
                    )
                )
        # aspect_ratio is hinted in the prompt; no direct API parameter is available.
        response = _call_gemini_api_with_retry(
            client=client,
            model=model,
            contents=parts,
            config=genai.types.GenerateContentConfig(
                seed=seed_value,
                system_instruction=SYSTEM_PROMPT,
                response_modalities=["TEXT", "IMAGE"] if response_modalities != "IMAGE" else ["IMAGE"],
            ),
        )

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
        ratio_hint = _aspect_ratio_hint(aspect_ratio, images)
        prompt_text = f"{prompt}\nAspect ratio: {ratio_hint}" if ratio_hint else prompt
        parts = [prompt_text]
        image_config = None
        image_config_args: dict[str, str] = {}
        if ratio_hint:
            image_config_args["aspect_ratio"] = ratio_hint
        if resolution:
            image_config_args["image_size"] = resolution
        if image_config_args:
            image_config = genai.types.ImageConfig(**image_config_args)
        if images is not None:
            # 支持多张图片输入
            png_bytes_list = _tensor_to_png_bytes_list(images)
            for png_bytes in png_bytes_list:
                parts.append(
                    genai.types.Part.from_bytes(
                        data=png_bytes,
                        mime_type="image/png",
                    )
                )
        response = _call_gemini_api_with_retry(
            client=client,
            model=model,
            contents=parts,
            config=genai.types.GenerateContentConfig(
                seed=seed_value,
                system_instruction=SYSTEM_PROMPT,
                image_config=image_config,
                response_modalities=["TEXT", "IMAGE"] if response_modalities != "IMAGE" else ["IMAGE"],
            ),
        )

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
