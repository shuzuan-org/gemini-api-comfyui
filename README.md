# Gemini API 节点（ComfyUI）

基于公开 Gemini API 的图像生成/编辑节点（不依赖 Vertex 专用接口）。节点同步执行，ComfyUI 不显示进度/流式。

## 安装
在 ComfyUI 根目录（或此目录）执行：
```
pip install -r custom_nodes/gemini-api-comfyui/requirements.txt
```

## 认证方式
- API Key：设置环境变量 `GEMINI_API_KEY=你的key`，或在 `custom_nodes/gemini-api-comfyui/` 下创建 `gemini_api_key.txt`，文件只写一行 key。
- Vertex（可选）：设置 `GEMINI_VERTEX_PROJECT=<项目ID>`（或 `GOOGLE_CLOUD_PROJECT`），可选 `GEMINI_VERTEX_LOCATION`（默认 `us-central1`），以及 `GOOGLE_APPLICATION_CREDENTIALS` 指向具备 Vertex 权限的 service account JSON。有项目 ID 时将使用 Vertex 模式，否则走 API Key 模式。

## 节点
- **Nano Banana (Google Gemini Image)**（`GeminiImageNanoBanana`）：文生图/可选参考图。模型：`gemini-2.5-flash-image`、`gemini-2.5-flash-image-preview`。种子 0–0x7FFFFFFF（负数/留空为随机）。`aspect_ratio` 含 `auto`。`response_modalities` 可选 仅图 / 图+文。
- **Nano Banana Pro (Google Gemini Image)**（`GeminiImageNanoBananaPro`）：同上，额外分辨率下拉（1K/2K/4K）。模型：`gemini-3-pro-image-preview`。

### 说明
- `aspect_ratio` 由公开 API 处理，`auto` 让服务自行决定。
+- 种子按 32 位限制（超出会被截断）。
+- `response_modalities` 选 `IMAGE` 时文本输出为空。
