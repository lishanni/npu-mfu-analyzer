# NPU MFU Analyzer

昇腾 NPU 大模型训练 MFU 智能分析工具

## 简介

NPU MFU Analyzer 是一个基于 Multi-Agent 架构的性能分析工具，用于分析昇腾 NPU 上大模型训练的 MFU（Model FLOPS Utilization）。

### 主要功能

| 功能 | 说明 |
|------|------|
| **Timeline 分析** | 分析计算/通信/空闲时间分布 |
| **Overlap 分析** | 分析 HCCL 通信与计算的重叠关系 |
| **MFU 计算** | 计算整体和算子级 MFU |
| **慢卡检测** | 识别集群中的慢卡问题 |
| **PP Bubble 分析** | Pipeline Parallel 气泡时间分析 |
| **通信拆分** | TP/DP/PP/CP/EP 通信开销拆分 |
| **智能建议** | 基于 LLM 生成优化建议 |
| **Web 界面** | 可视化分析界面 |

## 安装

### 前置条件

- Python >= 3.9, < 3.15
- pip

### 安装步骤

```bash
# 1. 克隆仓库
cd /path/to/npu-mfu-analyzer

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 3. 安装基础依赖
pip install -e .

# 4. 安装 Web 依赖（可选，用于 Web 界面）
pip install -e ".[web]"
# 或手动安装
pip install fastapi uvicorn python-multipart websockets aiofiles

# 5. 配置 LLM API Key（可选，使用 mock 后端则不需要）
export OPENAI_API_KEY="your-api-key"
# 或
export ANTHROPIC_API_KEY="your-api-key"
```

## 使用方式

### 方式一：命令行（CLI）

#### 查看 Profiling 数据信息

```bash
npu-analyzer info /path/to/profiling
```

输出示例：
```
数据类型: db
框架: pytorch
Rank 数量: 8
Timeline 数据: ✓
Memory 数据: ✓
Communication 数据: ✓
```

#### 生成数据摘要（不调用 LLM）

```bash
npu-analyzer summary /path/to/profiling
```

#### 执行完整分析

```bash
# 使用 Mock 后端（测试/无需 API Key）
npu-analyzer analyze /path/to/profiling --backend mock

# 使用 OpenAI
npu-analyzer analyze /path/to/profiling --backend openai

# 使用 Claude
npu-analyzer analyze /path/to/profiling --backend claude

# 指定输出文件
npu-analyzer analyze /path/to/profiling -o report.md

# 详细输出模式
npu-analyzer analyze /path/to/profiling -v
```

### 方式二：Web 界面

#### 启动 Web 服务

```bash
# 默认端口 8000
npu-analyzer web

# 指定端口
npu-analyzer web --port 8080

# 开发模式（自动重载）
npu-analyzer web --reload
```

#### 访问

- **Web 界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc

#### Web 界面功能

1. **分析表单**：输入 Profiling 路径，选择 LLM 后端和输出格式
2. **实时进度**：WebSocket 推送分析进度
3. **报告预览**：在线预览 HTML/Markdown 报告
4. **历史任务**：查看和管理历史分析任务

### 方式三：Python API

```python
import asyncio
from src.llm.llm_interface import LLMConfig
from src.agents.orchestrator import Orchestrator
from src.report.report_generator import ReportFormat

async def main():
    # 创建 Orchestrator
    config = LLMConfig(backend="mock")  # 或 "openai", "claude"
    orchestrator = Orchestrator("/path/to/profiling", llm_config=config)
    
    # 执行分析
    report = await orchestrator.run(output_format=ReportFormat.HTML)
    
    # 输出报告
    if report.success:
        print(report.final_report)
        print(f"优化建议: {report.recommendations}")
    else:
        print(f"分析失败: {report.error}")

asyncio.run(main())
```

## 支持的数据格式

### DB 格式（推荐）
- `ascend_pytorch_profiler_*.db`
- `analysis.db`
- `cluster_analysis.db`

### JSON 格式
- `trace_view.json`（支持流式解析大文件）
- `communication.json`

### CSV 格式（降级）
- `step_trace_time.csv`（当 DB 无 STEP_TRACE 表时自动使用）
- `kernel_details.csv`
- `operator_details.csv`

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传 Profiling 数据 |
| POST | `/api/analyze` | 启动分析任务 |
| GET | `/api/tasks` | 获取任务列表 |
| GET | `/api/tasks/{id}` | 查询任务状态 |
| GET | `/api/reports/{id}` | 获取分析报告 |
| DELETE | `/api/tasks/{id}` | 删除任务 |
| WS | `/ws/{id}` | WebSocket 实时进度 |

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `ANTHROPIC_API_KEY` | Claude API Key | - |
| `NPU_ANALYZER_DATA_DIR` | 数据存储目录 | `./data` |

### 配置文件

`config/settings.yaml`：

```yaml
llm:
  backend: openai  # openai / claude / mock
  openai:
    model: gpt-4-turbo-preview
    temperature: 0.1
  claude:
    model: claude-3-opus-20240229

data:
  streaming:
    enabled: true
  prefer_db: true

analysis:
  overlap:
    enabled: true
  slow_rank:
    enabled: true
  bubble:
    enabled: true
```

## 开发

### 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### 代码检查

```bash
ruff check src/
mypy src/
```

## 架构

```
src/
├── agents/           # Multi-Agent 核心
│   ├── base_agent.py       # Agent 基类
│   ├── orchestrator.py     # 编排器
│   ├── timeline_agent.py   # Timeline 分析
│   ├── operator_agent.py   # 算子分析
│   ├── memory_agent.py     # 内存分析
│   ├── communication_agent.py  # 通信分析
│   └── advisor_agent.py    # 综合建议
├── analyzers/        # 分析算法
│   ├── overlap_calculator.py   # Overlap 计算
│   ├── slow_rank_detector.py   # 慢卡检测
│   ├── bubble_analyzer.py      # PP Bubble 分析
│   ├── comm_splitter.py        # 通信拆分
│   └── mfu_calculator.py       # MFU 计算
├── data_loader/      # 数据加载层
│   ├── profiling_loader.py     # 数据加载器
│   ├── stream_parser.py        # 流式解析
│   └── data_summarizer.py      # 数据摘要
├── llm/              # LLM 接口层
│   ├── llm_interface.py        # LLM 接口
│   └── prompts/                # System Prompts
├── report/           # 报告生成
│   ├── report_generator.py     # 报告生成器
│   └── templates.py            # 报告模板
├── cli/              # 命令行接口
│   └── main.py
└── web/              # Web 界面
    ├── app.py                  # FastAPI 应用
    └── static/                 # 前端静态文件
```

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 0.1.0 | 2026-01 | 初始版本，支持 Timeline/Operator/Memory/Communication 分析 |

## License

Apache-2.0
