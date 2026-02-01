# 配置说明

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OLLAMA_HOST` | Ollama 服务地址 | `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `ANTHROPIC_API_KEY` | Claude API Key | - |
| `NPU_ANALYZER_DATA_DIR` | 数据存储目录 | `./data` |

## 配置文件

配置文件路径: `config/settings.yaml`

### 完整配置示例

```yaml
# LLM 配置
llm:
  # 后端选择: ollama / deepseek / openai / claude / mock
  backend: ollama
  
  ollama:
    model: qwen2.5:7b
    host: http://localhost:11434
    timeout: 120
  
  deepseek:
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com
  
  openai:
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
    temperature: 0.1
  
  claude:
    model: claude-3-opus-20240229
    api_key: ${ANTHROPIC_API_KEY}

# 硬件配置
hardware:
  auto_detect: true           # 自动从 profiling 数据检测硬件
  default: atlas_a2_280t      # 默认硬件型号
  
# 数据加载配置
data:
  streaming:
    enabled: true             # 启用流式解析大文件
    chunk_size: 10000         # 流式解析块大小
  prefer_db: true             # 优先使用 DB 格式
  fallback_csv: true          # DB 无数据时回退到 CSV

# 分析配置
analysis:
  overlap:
    enabled: true
  slow_rank:
    enabled: true
    method: three_sigma       # three_sigma / dixon_q / both
    threshold: 2.0            # sigma 阈值
  bubble:
    enabled: true
  jitter:
    enabled: true
    cv_threshold: 0.15        # 变异系数阈值
  roofline:
    enabled: true
  whatif:
    enabled: true

# Web 服务配置
web:
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - http://localhost:3000
    - http://localhost:8000

# 报告配置
report:
  default_format: html        # html / markdown / json
  include_raw_data: false
  max_suggestions: 10
```

## LLM 后端配置详解

### Ollama (推荐本地部署)

```yaml
llm:
  backend: ollama
  ollama:
    model: qwen2.5:7b         # 推荐中文模型
    host: http://localhost:11434
    timeout: 120              # 超时时间（秒）
```

支持的模型:
- `qwen2.5:7b` - 中文能力强，推荐
- `qwen2.5:14b` - 更强能力，需要更多显存
- `llama3.1:8b` - 英文为主
- `codellama:7b` - 代码相关

### DeepSeek API

```yaml
llm:
  backend: deepseek
  deepseek:
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com
    max_tokens: 4096
```

### OpenAI API

```yaml
llm:
  backend: openai
  openai:
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
    temperature: 0.1
    max_tokens: 4096
```

### Mock (测试用)

```yaml
llm:
  backend: mock               # 不调用真实 LLM，返回模拟响应
```

## 支持的数据格式

### DB 格式（推荐）

| 文件 | 说明 |
|------|------|
| `ascend_pytorch_profiler_*.db` | 主 profiling 数据库 |
| `analysis.db` | 分析结果数据库 |
| `cluster_analysis.db` | 多卡聚合数据库 |

### JSON 格式

| 文件 | 说明 |
|------|------|
| `trace_view.json` | Timeline 数据（支持 GB 级流式解析） |
| `communication.json` | 通信数据 |

### CSV 格式（降级）

| 文件 | 说明 |
|------|------|
| `step_trace_time.csv` | Step 时间统计 |
| `kernel_details.csv` | Kernel 详情 |
| `operator_details.csv` | 算子详情 |

## 命令行参数

### npu-analyzer analyze

```bash
npu-analyzer analyze <profiling_path> [OPTIONS]

Options:
  --backend TEXT      LLM 后端 (ollama/deepseek/openai/mock)
  --model TEXT        LLM 模型名称
  -o, --output TEXT   输出文件路径
  --format TEXT       输出格式 (html/markdown/json)
  -v, --verbose       详细输出
  --no-llm            不使用 LLM，仅生成数据摘要
```

### npu-analyzer web

```bash
npu-analyzer web [OPTIONS]

Options:
  --host TEXT    服务地址 (default: 0.0.0.0)
  --port INT     服务端口 (default: 8000)
  --reload       开发模式，自动重载
```
