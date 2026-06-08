# NPU MFU Analyzer

面向昇腾 NPU 大模型训练 profiling 的 MFU 与性能瓶颈诊断工具。

[English](README.md) | [安装指南](docs/installation.md) | [配置说明](docs/configuration.md) | [API 文档](docs/api_reference.md) | [架构说明](docs/architecture.md)

NPU MFU Analyzer 不是替代 profiler，而是把 profiling 数据翻译成“主矛盾、证据链和下一步实验”。它适合 AI Infra、训练性能优化和集群运维同学，用来解释低 MFU、通信暴露、空闲时间过长、pipeline bubble、overlap 失败等问题。

## 为什么需要它

msprof、Ascend Profiler 和 PyTorch Profiler 擅长采集原始数据，但真实大模型训练问题通常还需要回答：

- 低 MFU 主要来自计算、通信、内存、pipeline 调度，还是 Host 输入间隙？
- 通信耗时高是 overlap 没藏住、通信过碎，还是拓扑/链路问题？
- 下一步应该优先做哪个实验，而不是把所有开关都试一遍？

这个项目提供 profiling 之上的诊断层，输出结构化瓶颈判断和可执行优化建议。

## 快速开始

```bash
git clone https://github.com/lishanni/npu-mfu-analyzer.git
cd npu-mfu-analyzer
pip install -e ".[dev]"

npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o report.md
```

样例输出会包含类似信息：

```text
Rank 数量: 1
平均 Step 时间: 31385.19 ms
计算: 2.0%
通信: 10.4%
空闲: 87.8%
主矛盾: INPUT_HOST_GAP
优先动作: 检查数据加载和 Host 调度
```

## LLM 后端

支持 `openai`、`claude`、`deepseek`、`ollama` 和 `mock`。`mock` 后端适合 CI 和本地 smoke test。

使用智谱 GLM 的 Anthropic 兼容接口：

```bash
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_api_key"
npu-analyzer analyze examples/sample_profiling_small -b claude -m GLM-4.7
```

不要把 API key 或内部真实 profiling 数据提交到仓库。

## 核心能力

| 能力 | 诊断内容 |
| --- | --- |
| MFU / Roofline 分析 | 硬件利用率低、计算效率不足 |
| Timeline 分析 | 计算、通信、空闲、overlap 拆解 |
| 通信分析 | 通信暴露、overlap 失败、慢链路、通信拆分 |
| Skills v2 诊断 | 主矛盾、训练场景、优先动作、what-if 实验 |
| Host-Device 关联 | 可选的调用栈/来源归因、输入间隙和同步问题 |
| AIC 微架构分析 | Cube/Vector 利用率、流水线停顿、内存层次问题 |
| Profiling 对比 | 两次 profiling 的性能回退诊断 |

## Host-Device 大 trace 保护

大型 `trace_view.json` 可能达到数 GB。为了保证首次运行体验，默认当 trace 超过 512 MB 时会自动跳过 Host-Device correlation。

```bash
# 提高阈值
npu-analyzer analyze ./profiling --host-device-max-trace-mb 2048

# 强制执行完整 Host-Device 关联
npu-analyzer analyze ./profiling --full-host-device-correlation

# 显式禁用 Host-Device 关联
npu-analyzer analyze ./profiling --no-host-device-correlation
```

跳过 Host-Device correlation 时，核心 MFU、timeline、通信、skills 和报告生成仍会继续执行。

## Python API

```python
import asyncio

from npu_mfu_analyzer.agents.orchestrator import Orchestrator
from npu_mfu_analyzer.llm.llm_interface import LLMConfig


async def main():
    config = LLMConfig(backend="mock")
    orchestrator = Orchestrator("examples/sample_profiling_small", llm_config=config)
    report = await orchestrator.run()
    print(report.final_report)


asyncio.run(main())
```

## 已知限制

- 大 trace 的 Host-Device correlation 默认受保护；需要全量分析时显式使用 `--full-host-device-correlation`。
- 不同 CANN/export 配置可能缺少部分 profiling 表，工具会尽量降级。
- 当前主要面向 Ascend NPU、PyTorch/torch-npu 和大模型训练任务。
- 仓库内样例数据是极小 smoke fixture，不是完整 benchmark。

## 推荐 GitHub Topics

`ascend`, `npu`, `mfu`, `profiling`, `llm-training`, `performance-analysis`, `torch-npu`, `mindspeed`, `distributed-training`

## 贡献

欢迎贡献。请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，提交 issue 时尽量提供脱敏后的 profiling 背景，不要上传密钥或未脱敏的内部 trace。

## License

Apache-2.0。请以 GitHub 仓库 license 元数据为准。

