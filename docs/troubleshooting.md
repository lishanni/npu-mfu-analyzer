# 故障排查

## 1. 数据加载问题

### 找不到 Profiling 数据

**症状**：`ProfilingLoader` 报错 "No profiling data found"

**排查**：
```bash
# 检查目录结构
ls -la /path/to/profiling/
# 应包含以下文件之一：
#   *.db (SQLite 数据库)
#   trace_view.json (Timeline 数据)
#   step_trace_time.csv (降级数据)
```

**解决方案**：
- 确保路径指向 msprof 输出的 profiling 目录
- 如果是多卡采集，路径应指向包含 `PROF_*` 子目录的父目录
- 检查是否有 rank/worker 子目录：`profiler_*`、`rank_*`、`worker_*`

### 大文件解析内存溢出

**症状**：处理 GB 级 `trace_view.json` 时内存不足

**解决方案**：
```yaml
# config/settings.yaml
data:
  streaming:
    enabled: true        # 启用流式解析（默认开启）
    chunk_size: 10000    # 调小块大小以减少内存
  prefer_db: true        # 优先使用 DB 格式（内存效率更高）
```

- 优先使用 `.db` 格式数据（支持索引查询，内存占用极低）
- 确保 `ijson` 已安装：`pip install ijson>=3.2.0`
- 5GB JSON 文件通常内存占用 < 2GB

### Decimal 类型转换错误

**症状**：`TypeError: Object of type Decimal is not JSON serializable`

**原因**：ijson 流式解析返回 `decimal.Decimal` 类型而非 `float`

**解决方案**：系统已内置修复。如仍遇到，确认使用最新版本：
```bash
git pull origin master
pip install -e ".[all]" --upgrade
```

### DB 格式不兼容

**症状**：`sqlite3.OperationalError: no such table: STEP_TRACE`

**解决方案**：
- 系统会自动降级到 CSV 文件（`step_trace_time.csv`）
- 确认 `config/settings.yaml` 中 `data.fallback_csv: true`
- 不同版本的 msprof 可能产生不同的表名，系统支持多种表名匹配

## 2. LLM 连接问题

### OpenAI API 连接失败

**症状**：`openai.APIConnectionError` 或超时

**排查**：
```bash
# 检查 API Key
echo $OPENAI_API_KEY

# 测试连通性
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**解决方案**：
- 检查环境变量是否正确设置
- 检查网络代理设置
- 使用弹性 LLM 自动降级：系统会自动 重试 3 次 → 切换到 DeepSeek → Ollama → Mock

### Ollama 本地模型连接失败

**症状**：`ConnectionError: Cannot connect to http://localhost:11434`

**排查**：
```bash
# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# 检查是否有模型
ollama list
```

**解决方案**：
```bash
# 启动 Ollama 服务
ollama serve

# 如果换了端口
export OLLAMA_HOST="http://localhost:PORT"
```

### Claude API 报错 "Invalid API Key"

**排查**：
```bash
# 检查 API Key
echo $ANTHROPIC_API_KEY

# 使用兼容 API 时检查 Base URL
echo $ANTHROPIC_BASE_URL
```

**解决方案**：
- 确保 API Key 格式正确（以 `sk-ant-` 开头）
- 兼容 API（如 GLM）需同时设置 `ANTHROPIC_BASE_URL`
- 检查 API Key 是否过期或额度不足

### LLM 响应超时

**症状**：分析过程中长时间无响应

**解决方案**：
```python
# 增加超时配置
from npu_mfu_analyzer.llm.resilient_llm import ResilientLLM, ResilientConfig, TimeoutConfig

config = ResilientConfig(
    timeout=TimeoutConfig(
        request_timeout=180.0,  # 增加到 3 分钟
        total_timeout=600.0,    # 增加到 10 分钟
    ),
)
```

或使用 Mock 模式跳过 LLM：
```bash
npu-analyzer analyze /path/to/profiling -b mock
```

## 3. 分析异常

### 硬件型号未识别

**症状**：`Unknown hardware: xxx`

**解决方案**：
- 系统会使用默认硬件规格（Atlas A2 280T）
- 可在 `config/settings.yaml` 中手动指定：
```yaml
hardware:
  auto_detect: false
  default: atlas_a2_376t
```
- 如需添加新硬件，参考 `src/hardware/specs/` 下的 YAML 文件

### 框架/并行策略识别错误

**症状**：Pattern Matcher 识别的框架或并行策略不正确

**排查**：
```bash
# 先查看识别结果
npu-analyzer info /path/to/profiling
```

**原因**：
- 自定义框架或非标准算子命名
- 通信组名称不符合常见模式

**解决方案**：
- 识别结果仅影响建议内容的针对性，不影响核心指标分析
- 可忽略不准确的识别结果

### Rank/Worker 识别失败

**症状**：多卡 Profiling 只分析了一张卡

**排查**：
- 检查目录结构是否包含正确的子目录命名
- 支持的目录命名模式：`rank_N`、`worker_N`、`profiler_N_*`

### 对比分析报告 "不建议对比"

**症状**：`compare` 命令返回 "两次 Profiling 没有直接关系"

**原因**：`SimilarityChecker` 评分 < 0.3

**解决方案**：
```bash
# 使用 --force 强制对比（如有意对比不同配置）
npu-analyzer compare /path/a /path/b --force
```

## 4. Web 界面问题

### Web 服务启动失败

**症状**：`ModuleNotFoundError: No module named 'fastapi'`

**解决方案**：
```bash
pip install -e ".[web]"
```

### WebSocket 连接断开

**症状**：Web 界面进度条停止更新

**排查**：
- 检查浏览器控制台 (F12) 中的 WebSocket 错误
- 长时间分析（>5 分钟）可能导致连接超时

**解决方案**：
- 刷新页面重新连接
- 分析结果会持久化，可通过 `/api/tasks/{task_id}` 查询

### 端口被占用

**症状**：`Address already in use`

**解决方案**：
```bash
# 使用其他端口
npu-analyzer web --port 8001

# 或查找并终止占用进程
lsof -i :8000
kill <PID>
```

## 5. 测试问题

### 单元测试失败

**排查**：
```bash
# 运行特定测试文件
pytest tests/unit/test_comparison.py -v

# 查看详细错误
pytest tests/unit/ -v --tb=long
```

**常见原因**：
- `ijson` 未安装导致 `test_data_loader.py` 失败
- `pandas` 版本不兼容导致 DataFrame 操作异常

**解决方案**：
```bash
pip install -e ".[dev,all]" --upgrade
```

## 6. 性能问题

### 分析速度慢

**可能原因与优化**：

| 原因 | 解决方案 |
|------|---------|
| 数据文件过大（>5GB JSON） | 转换为 DB 格式或确认启用 streaming |
| LLM 响应慢 | 使用本地 Ollama 或 Mock 模式 |
| 多 Agent 串行执行 | 系统默认并行执行，检查日志确认 |
| 网络延迟（远程 API） | 使用本地模型或增加超时 |

### 减少分析时间

```bash
# 减少采样 Step 数
npu-analyzer summary /path/to/profiling --max-steps 5

# 使用 Mock 模式（仅数据分析，秒级完成）
npu-analyzer analyze /path/to/profiling -b mock

# 使用本地 Ollama（避免网络延迟）
npu-analyzer analyze /path/to/profiling -b ollama
```
