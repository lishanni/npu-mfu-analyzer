# 配置说明

## 配置优先级

配置项按以下优先级生效（高 → 低）：

1. **CLI 参数**：`--backend openai` 等命令行选项
2. **环境变量**：`OPENAI_API_KEY` 等
3. **配置文件**：`config/settings.yaml`
4. **默认值**：代码内置默认值

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OLLAMA_HOST` | Ollama 服务地址 | `http://localhost:11434` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `ANTHROPIC_API_KEY` | Claude API Key | - |
| `ANTHROPIC_BASE_URL` | Claude API 自定义地址（用于代理或兼容API） | - |
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

### Claude API

```yaml
llm:
  backend: claude
  claude:
    model: claude-3-opus-20240229
    api_key: ${ANTHROPIC_API_KEY}
    base_url: ${ANTHROPIC_BASE_URL}  # 可选，支持自定义endpoint或代理
    max_tokens: 4096
```

**使用兼容API（如智谱GLM）**：
```bash
# 设置环境变量
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_api_key"

# 使用
npu-analyzer analyze /path/to/profiling --backend claude -m GLM-4.7
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

## ResilientLLM 配置

弹性 LLM 客户端支持额外的容错配置：

```yaml
llm:
  resilient:
    retry:
      max_retries: 3          # 最大重试次数
      base_delay: 1.0         # 基础延迟（秒）
      max_delay: 30.0         # 最大延迟（秒）
      exponential_base: 2.0   # 指数退避基数
    fallback:
      enabled: true
      fallback_backends:      # 降级后端顺序
        - deepseek
        - ollama
        - mock
    timeout:
      request_timeout: 120.0  # 单次请求超时（秒）
      total_timeout: 300.0    # 总超时（秒）
```

## 命令行参数

> 完整 CLI 和 API 参考请见 [API 参考文档](api_reference.md)

### npu-analyzer analyze

```bash
npu-analyzer analyze <profiling_path> [OPTIONS]

Options:
  -b, --backend TEXT    LLM 后端 (openai/claude/ollama/deepseek/mock) [default: openai]
  -m, --model TEXT      LLM 模型名称
  -o, --output TEXT     输出文件路径
  -f, --format TEXT     输出格式 (markdown/html/md) [default: markdown]
  -v, --verbose         详细输出
```

### npu-analyzer compare

```bash
npu-analyzer compare <PATH_A> <PATH_B> [OPTIONS]

Options:
  -b, --backend TEXT    LLM 后端 [default: mock]
  -m, --model TEXT      LLM 模型名称
  -o, --output TEXT     输出报告路径
  -f, --format TEXT     报告格式 (markdown/html/md) [default: markdown]
  --label-a TEXT        基准版本标签
  --label-b TEXT        当前版本标签
  --force               跳过相似度检查，强制对比
  -v, --verbose         详细输出
```

### npu-analyzer generate

```bash
npu-analyzer generate <profiling_path> [OPTIONS]

Options:
  -b, --backend TEXT          LLM 后端 [default: claude]
  -m, --model TEXT            LLM 模型名称
  -o, --output TEXT           输出目录 [default: ./generated_kernels]
  --min-speedup FLOAT         最小加速比阈值 [default: 1.05]
  --complexity TEXT            最大实现复杂度 (低/中等/高) [default: 高]
  --skip-native/--include-native  跳过/包含昇腾原生融合算子
  --timeout INT               单个算子生成超时（秒） [default: 300]
  --max-concurrent INT        最大并发生成数 [default: 3]
  -v, --verbose               详细输出
```

### npu-analyzer integrate

```bash
npu-analyzer integrate <profiling_path> [OPTIONS]

Options:
  -o, --output TEXT     输出目录 [default: ./integration_output]
  --patterns TEXT       融合模式列表，逗号分隔 [default: add,mul,slice,strided]
  --time-window INT     时间窗口（微秒） [default: 100]
  --limit INT           分析的最大调用数 [default: 50]
  -v, --verbose         详细输出
```

### npu-analyzer analyze-aic

```bash
npu-analyzer analyze-aic <profiling_path> [OPTIONS]

Options:
  -n, --top-n INT       显示 Top N 算子 [default: 20]
  -s, --sort-by TEXT    排序方式 (duration/cube_util/l2_hit/stall_rate/name) [default: duration]
  -a, --show-all        显示所有指标
  -o, --output TEXT     输出文件（CSV 或 Markdown）
  --severity TEXT       按严重度筛选 (all/critical/high/medium/low) [default: all]
```

### npu-analyzer web

```bash
npu-analyzer web [OPTIONS]

Options:
  -h, --host TEXT    服务地址 [default: 0.0.0.0]
  -p, --port INT     服务端口 [default: 8000]
  --reload           开发模式，自动重载
```

### npu-analyzer info / summary / version

```bash
# 查看 Profiling 数据信息
npu-analyzer info <profiling_path>

# 生成数据摘要（不调用 LLM）
npu-analyzer summary <profiling_path> [--max-steps INT]

# 查看版本信息
npu-analyzer version
```
