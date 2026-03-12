# CLAUDE.md - AI Agent Guide for NPU MFU Analyzer

> This file helps AI agents (Claude, GPT, GitHub Copilot, etc.) understand and work with this project effectively.

## Project Overview

**NPU MFU Analyzer** is a professional performance analysis tool for LLM training on Huawei Ascend NPU. It provides end-to-end analysis from profiling data parsing to optimization recommendations.

### What This Tool Does

- **MFU Analysis**: Calculate Model Flops Utilization for training workloads
- **Performance Bottleneck Detection**: Identify compute, communication, memory bottlenecks
- **Profiling Comparison**: Compare two profiling runs and find root causes of regression
- **AIC Microarchitecture Analysis**: Diagnose AI Core bottlenecks (Cube/Vector utilization, L2 cache)
- **Fusion Operator Discovery**: Find operator fusion opportunities from profiling data
- **Root Cause Inference**: Automated root cause analysis with rule-based engine

### Target Users

- AI Infrastructure Engineers working with Ascend NPU
- LLM Training Engineers optimizing model performance
- Performance Analysts debugging training bottlenecks
- DevOps Engineers monitoring training clusters

### Supported Hardware

- Atlas A2 series: 280T, 313T, 376T
- Atlas 300I series
- More hardware support via extensible registry

## Key Technologies

| Category | Technologies |
|----------|-------------|
| **NPU** | Huawei Ascend, CANN, ATB, AIC, AIV |
| **Frameworks** | PyTorch, MindSpore, torch.compile, Megatron, DeepSpeed, FSDP |
| **Communication** | HCCL, HCCS, RDMA, NCCL-compatible |
| **LLM Backends** | OpenAI, Claude, DeepSeek, Ollama, Mock |

## Quick Integration Examples

### CLI Usage

```bash
# Analyze profiling data
npu-analyzer analyze /path/to/profiling --backend ollama

# Compare two profiling runs
npu-analyzer compare /path/to/before /path/to/after -b openai

# AIC hardware metrics analysis
npu-analyzer analyze-aic /path/to/profiling

# Generate fusion operator code
npu-analyzer generate /path/to/profiling -b claude
```

### Python API

```python
import asyncio
from src.agents.orchestrator import Orchestrator
from src.llm import LLMConfig

async def analyze():
    config = LLMConfig(backend="ollama", model="qwen2.5:7b")
    orchestrator = Orchestrator("/path/to/profiling", llm_config=config)
    report = await orchestrator.run()
    print(report.final_report)

asyncio.run(analyze())
```

## Architecture Summary

```
Presentation Layer (CLI/Web/Python API)
    ↓
Agent Layer (13 Specialized Agents)
    ↓
Skill Layer (14 Expert Skills)
    ↓
Analysis Layer (15+ Analyzers)
    ↓
Data Layer (Profiling Loader, Stream Parser, DB Query)
```

## Key Files for AI Agents

| File | Purpose |
|------|---------|
| `src/agents/orchestrator.py` | Main analysis orchestration |
| `src/analyzers/` | Core analysis engines |
| `src/data_loader/profiling_loader.py` | Unified data loading interface |
| `src/hardware/registry.py` | Hardware specifications |
| `src/llm/` | LLM backend implementations |
| `src/cli/` | Command-line interface |
| `docs/` | Detailed documentation |

## Common Tasks for AI Agents

### 1. Add New Analysis Feature

1. Create analyzer in `src/analyzers/`
2. Create corresponding agent in `src/agents/`
3. Register in orchestrator
4. Add CLI command in `src/cli/`

### 2. Add New Hardware Support

1. Add specs to `src/hardware/registry.py`
2. Implement detection logic
3. Update documentation

### 3. Add New LLM Backend

1. Implement backend in `src/llm/backends/`
2. Register in `src/llm/factory.py`
3. Add configuration support

### 4. Fix Bugs

- Follow existing code patterns
- Maintain type annotations
- Add tests for fixes
- Reference msprof implementation for profiling data handling

## Coding Conventions

- **Style**: ruff formatter, line-length=120
- **Types**: Full type annotations, dataclass preferred
- **Async**: Use asyncio for I/O operations
- **Error Handling**: Comprehensive exception handling with logging
- **Testing**: pytest with ~115 test cases

## Important Notes

### Profiling Data Handling

> **CRITICAL**: Always reference msprof (Huawei's official profiler) implementation when working with profiling data. Field mappings and calculation formulas must match msprof exactly.

Key field mappings:
- `Communication(Not Overlapped)` → `communication_not_overlapped`
- `Computing` → `computing_time`
- `Free` → `free_time`

Step time formula:
```python
step_time = Computing + Communication(Not Overlapped) + Free
```

### LLM Integration

- Supports multiple backends with automatic failover
- Mock backend available for testing without LLM
- ResilientLLM provides retry/timeout/fallback

## Resources

- [Documentation](docs/)
- [Installation Guide](docs/installation.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)

## Contact & Contribution

- GitHub: https://github.com/lishanni/npu-mfu-analyzer
- Issues: https://github.com/lishanni/npu-mfu-analyzer/issues
- License: Apache-2.0

---

*This file is optimized for AI agent understanding and integration.*
