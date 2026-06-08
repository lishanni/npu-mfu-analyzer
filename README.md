# NPU MFU Analyzer

Diagnose Ascend NPU LLM training bottlenecks from profiling traces.

[中文说明](README.zh-CN.md) | [Installation](docs/installation.md) | [Configuration](docs/configuration.md) | [API Reference](docs/api_reference.md) | [Architecture](docs/architecture.md)

NPU MFU Analyzer turns Ascend profiling output into a structured diagnosis: where time goes, what the main bottleneck is, and which optimization experiments to try first. It is built for AI infrastructure and LLM training engineers who need to explain low MFU, communication exposure, idle time, pipeline bubbles, and overlap failures.

## Why It Exists

Raw profiling tools are excellent at collecting data, but production LLM training problems often need a higher-level answer:

- Is low MFU caused by compute, communication, memory, pipeline scheduling, or Host input gaps?
- Is communication expensive because overlap failed, collectives are fragmented, or topology is weak?
- Which experiment should be tried first instead of tuning everything at once?

This project is not a replacement for msprof, Ascend Profiler, or PyTorch Profiler. It is a diagnosis layer on top of profiling data.

## Quick Start

```bash
git clone https://github.com/lishanni/npu-mfu-analyzer.git
cd npu-mfu-analyzer
pip install -e ".[dev]"

npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o report.md
```

Expected sample output:

```text
Rank count: 1
Average step time: 31385.19 ms
Compute: 2.0%
Communication: 10.4%
Idle: 87.8%
Main contradiction: INPUT_HOST_GAP
Top action: inspect data loading and host scheduling
```

## LLM Backends

The analyzer supports `openai`, `claude`, `deepseek`, `ollama`, and `mock`. The `mock` backend is useful for CI and local smoke tests.

For Zhipu GLM through an Anthropic-compatible API:

```bash
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_api_key"
npu-analyzer analyze examples/sample_profiling_small -b claude -m GLM-4.7
```

Do not commit API keys or real internal profiling data to the repository.

## Core Capabilities

| Capability | What it helps diagnose |
| --- | --- |
| MFU and roofline analysis | Low hardware utilization and compute efficiency gaps |
| Timeline analysis | Compute, communication, free/idle, and overlap breakdown |
| Communication analysis | Exposed collectives, poor overlap, slow links, and communication split |
| Skills v2 diagnosis | Main contradiction, training scenario, prioritized actions, what-if experiments |
| Host-Device correlation | Optional stack/source attribution for input gaps and synchronization issues |
| AIC microarchitecture analysis | Cube/Vector utilization, pipeline stalls, and memory hierarchy issues |
| Comparison analysis | Regression diagnosis between two profiling runs |

## Host-Device Correlation Guardrail

Large `trace_view.json` files can be several GB. To keep first-run behavior predictable, Host-Device correlation is automatically skipped when the selected trace is larger than 512 MB.

```bash
# Keep the default guardrail but raise the threshold.
npu-analyzer analyze ./profiling --host-device-max-trace-mb 2048

# Force full Host-Device correlation for large traces.
npu-analyzer analyze ./profiling --full-host-device-correlation

# Disable Host-Device correlation explicitly.
npu-analyzer analyze ./profiling --no-host-device-correlation
```

Core MFU, timeline, communication, skills, and report generation still run when Host-Device correlation is skipped.

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

## Known Limitations

- Large trace Host-Device correlation is protected by default; opt in with `--full-host-device-correlation`.
- Some profiling tables may be missing depending on CANN/export configuration; the analyzer degrades where possible.
- The project currently focuses on Ascend NPU, PyTorch/torch-npu, and LLM training workloads.
- The bundled sample data is a tiny smoke fixture, not a complete benchmark.

## Recommended GitHub Topics

`ascend`, `npu`, `mfu`, `profiling`, `llm-training`, `performance-analysis`, `torch-npu`, `mindspeed`, `distributed-training`

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), open an issue with sanitized profiling context, and never attach secrets or proprietary traces without redaction.

## License

Apache-2.0. See [LICENSE](LICENSE) if present in your checkout, or the repository license metadata on GitHub.
