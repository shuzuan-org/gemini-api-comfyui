# Gemini API 节点（ComfyUI）

基于公开 Gemini API 的图像生成/编辑节点（不依赖 Vertex 专用接口）。节点同步执行，ComfyUI 不显示进度/流式。

## 安装
在 ComfyUI 根目录（或此目录）执行：
```
pip install -r custom_nodes/gemini-api-comfyui/requirements.txt
```

## 认证方式

### 官方 Gemini 节点（Nano Banana / Pro）
- **API Key**：设置环境变量 `GEMINI_API_KEY=你的key`，或在 `custom_nodes/gemini-api-comfyui/` 下创建 `gemini_api_key.txt`，文件只写一行 key。
- **Vertex（可选）**：设置 `GEMINI_VERTEX_PROJECT=<项目ID>`（或 `GOOGLE_CLOUD_PROJECT`），可选 `GEMINI_VERTEX_LOCATION`（默认 `us-central1`），以及 `GOOGLE_APPLICATION_CREDENTIALS` 指向具备 Vertex 权限的 service account JSON。有项目 ID 时将使用 Vertex 模式，否则走 API Key 模式。

### 代理节点（Gemini Image Proxy）
- **代理 API Key**（与上述 `GEMINI_API_KEY` **不是同一个**）：设置环境变量 `GEMINI_PROXY_API_KEY=你的代理key`，或在同目录下创建 `gemini_proxy_api_key.txt`，文件只写一行 key。仅 **Gemini Image (Proxy)** 节点使用此 key。

## 节点
- **Nano Banana (Google Gemini Image)**（`GeminiImageNanoBanana`）：文生图/可选参考图。模型：`gemini-2.5-flash-image`、`gemini-2.5-flash-image-preview`。种子 0–0x7FFFFFFF（负数/留空为随机）。`aspect_ratio` 含 `auto`。`response_modalities` 可选 仅图 / 图+文。
- **Nano Banana Pro (Google Gemini Image)**（`GeminiImageNanoBananaPro`）：同上，额外分辨率下拉（1K/2K/4K）。模型：`gemini-3-pro-image-preview`。
- **Gemini Image (Proxy)**（`GeminiImageProxy`）：通过**生成式 API 代理服务**调用 `generateImage` 接口。使用独立的代理 API Key（见上文）。支持 prompt、可选参考图、`aspect_ratio`（含 `auto`，无图时默认 16:9）、`image_size`（1K/2K/4K）、`response_modalities`、`auto_fallback`。默认超时 3 分钟，可通过 `GEMINI_PROXY_TIMEOUT` 调整。

### 说明
- `aspect_ratio` 由公开 API 处理；`auto` 时：有参考图则按首图比例取最接近的允许比例，无图时由服务或默认（Proxy 为 16:9）决定。
- 种子按 32 位限制（超出会被截断）；Proxy 节点无种子参数。
- `response_modalities` 选 `IMAGE` 时文本输出为空。

## 高级配置

### API调用优化配置
为了防止API调用失败导致的无限重试和资源耗尽，本插件实现了以下保护机制：

#### 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `GEMINI_API_TIMEOUT` | `600` | 官方 API 调用超时时间（秒，默认10分钟）<br/>**重要**：4K Pro模型可能需要5-10分钟 |
| `GEMINI_MAX_RETRIES` | `2` | 失败后的重试次数<br/>**总尝试次数 = 1（初始）+ 2（重试）= 3次** |
| `GEMINI_RETRY_DELAY` | `5.0` | 重试之间的延迟（秒） |
| `GEMINI_CB_THRESHOLD` | `5` | 断路器阈值：连续失败多少次后断路 |
| `GEMINI_CB_TIMEOUT` | `300` | 断路器超时：断路后等待多少秒再尝试恢复（默认5分钟） |
| **代理节点** | | |
| `GEMINI_PROXY_BASE_URL` | `https://gate.origintask.cn` | 生成式 API 代理服务根地址（仅 Proxy 节点） |
| `GEMINI_PROXY_TIMEOUT` | `180` | 代理 API 请求超时（秒，默认 3 分钟） |
| `GEMINI_PROXY_API_KEY` | — | 代理服务 API Key（也可用 `gemini_proxy_api_key.txt`） |

#### 预期生成时间

不同模型和配置的图像生成时间参考：

| 模型类型 | 分辨率 | 预期时间 | 备注 |
|---------|--------|---------|------|
| `gemini-2.5-flash-image` | 默认 | 30-120秒 | 快速模型，适合实时生成 |
| `gemini-3-pro-image-preview` | 1K | 60-150秒 | 标准质量 |
| `gemini-3-pro-image-preview` | 2K | 90-240秒 | 中等质量 |
| `gemini-3-pro-image-preview` | **4K** | **180-600秒** | 高质量，可能需要5-10分钟 |

**提示：** 复杂的提示词、多图参考、高峰时段都会增加生成时间。

#### 断路器模式
当API连续失败达到阈值（默认5次）时，断路器会自动打开，阻止后续请求，避免资源耗尽。断路器会在配置的超时时间后尝试恢复。

**断路器状态：**
- `CLOSED`：正常状态，允许请求
- `OPEN`：断路状态，拒绝所有请求
- `HALF_OPEN`：尝试恢复状态，允许一次测试请求

### 监控和日志

插件会输出详细的日志信息，包括：
- API调用成功/失败统计
- 请求耗时
- 成功率百分比
- 断路器状态变化
- 重试次数和原因

**成功调用日志示例：**
```
2025-12-16 10:30:00 - gemini_nodes - INFO - Gemini客户端已初始化，超时设置: 600秒
2025-12-16 10:30:01 - gemini_nodes - INFO - 调用Gemini API (初始尝试，总进度 1/3)，模型: gemini-3-pro-image-preview
2025-12-16 10:35:23 - gemini_nodes - INFO - API调用成功，耗时: 322.45秒
2025-12-16 10:35:23 - gemini_nodes - INFO - API调用成功，成功率: 98.5% (197/200)
```

**重试日志示例：**
```
2025-12-16 10:30:01 - gemini_nodes - INFO - 调用Gemini API (初始尝试，总进度 1/3)，模型: gemini-2.5-flash-image
2025-12-16 10:35:00 - gemini_nodes - WARNING - API调用失败 (尝试 1/3)，耗时: 299.12秒，错误: Request timeout。5.0秒后进行第1次重试...
2025-12-16 10:35:05 - gemini_nodes - INFO - 调用Gemini API (第1次重试，总进度 2/3)，模型: gemini-2.5-flash-image
2025-12-16 10:36:15 - gemini_nodes - INFO - API调用成功，耗时: 70.23秒
```

**断路器打开日志示例：**
```
2025-12-16 10:35:30 - gemini_nodes - ERROR - 连续失败 5 次，断路器打开！总请求数: 205，成功率: 95.1%。将在 300 秒后尝试恢复
2025-12-16 10:36:00 - gemini_nodes - ERROR - 断路器处于OPEN状态，拒绝请求。失败次数: 5/5，总请求数: 206，成功率: 95.1%，将在 270秒 后重试
```

## 故障排除

### 生成超时问题
如果4K或复杂图像经常超时：

```bash
# 增加超时时间到15分钟（针对4K Pro模型）
export GEMINI_API_TIMEOUT=900

# 或者更长（20分钟）
export GEMINI_API_TIMEOUT=1200
```

**注意：** 对于Flash模型，默认600秒已经足够。只有Pro模型的4K图像才需要更长时间。

### CPU占用过高
如果遇到CPU占用持续过高的问题：

1. **检查日志**：查看是否有大量失败请求或重试
2. **调整断路器阈值**：降低 `GEMINI_CB_THRESHOLD`（如设为3）
3. **检查超时设置**：确认超时时间足够长（4K至少600秒）
4. **增加重试延迟**：提高 `GEMINI_RETRY_DELAY`（如设为10.0）
5. **检查网络连接**：确保能够稳定访问Gemini API
6. **验证API密钥**：确保API密钥有效且未过期

### 子进程异常退出
如果遇到子进程不断退出（exit code 255）：

1. **检查依赖**：确保 `google-genai` 包已正确安装
2. **验证认证**：确认API密钥或Vertex配置正确
3. **查看日志**：检查详细错误信息
4. **重启服务**：重启ComfyUI进程

### 断路器频繁打开
如果断路器频繁打开：

1. **增加阈值**：提高 `GEMINI_CB_THRESHOLD`（如设为10）
2. **检查网络**：确保网络连接稳定
3. **检查配额**：确认API配额未超限
4. **联系支持**：如果问题持续，联系Gemini API支持
