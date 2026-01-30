# NPU MFU Analyzer

昇腾 NPU 大模型训练 MFU 智能分析工具

## 简介

NPU MFU Analyzer 是一个基于 Multi-Agent 架构的性能分析工具，用于分析昇腾 NPU 上大模型训练的 MFU（Model FLOPS Utilization）。

主要功能：
- **Timeline 分析**：分析计算/通信/空闲时间分布
- **Overlap 分析**：分析 HCCL 通信与计算的重叠关系
- **MFU 计算**：计算整体和算子级 MFU
- **慢卡检测**：识别集群中的慢卡问题
- **智能建议**：基于 LLM 生成优化建议

## 安装

### 前置条件

- Python >= 3.9, < 3.12
- pip

### 安装步骤

```bash
# 1. 克隆仓库
cd /path/to/npu-mfu-analyzer

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -e .

# 4. 配置 LLM API Key
export OPENAI_API_KEY="your-api-key"
```

## 快速开始

### 查看 Profiling 数据信息

```bash
npu-analyzer info /path/to/profiling
```

### 生成数据摘要（不调用 LLM）

```bash
npu-analyzer summary /path/to/profiling
```

### 执行完整分析

```bash
# 使用 OpenAI
npu-analyzer analyze /path/to/profiling

# 指定输出文件
npu-analyzer analyze /path/to/profiling -o report.md

# 使用 Mock 后端（测试）
npu-analyzer analyze /path/to/profiling --backend mock

# 详细输出
npu-analyzer analyze /path/to/profiling -v
```

## 支持的数据格式

### DB 格式（推荐）
- `ascend_pytorch_profiler_*.db`
- `analysis.db`
- `cluster_analysis.db`

### JSON 格式
- `trace_view.json`
- `communication.json`

## 配置

配置文件位于 `config/settings.yaml`：

```yaml
llm:
  backend: openai
  openai:
    model: gpt-4-turbo-preview
    temperature: 0.1

data:
  streaming:
    enabled: true
  prefer_db: true

analysis:
  overlap:
    enabled: true
  slow_rank:
    enabled: true
```

## 开发

### 运行测试

```bash
pip install -e ".[dev]"
pytest tests/
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
│   ├── base_agent.py
│   ├── orchestrator.py
│   └── ...
├── data_loader/      # 数据加载层
│   ├── profiling_loader.py
│   ├── stream_parser.py
│   └── data_summarizer.py
├── analyzers/        # 分析算法
├── llm/              # LLM 接口层
│   ├── llm_interface.py
│   └── prompts/
├── cli/              # 命令行接口
└── web/              # Web 界面（Phase 4）
```

## License

Apache-2.0
