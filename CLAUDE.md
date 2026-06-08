# CLAUDE.md - AI Agent Guide for NPU MFU Analyzer

This file helps Claude, GPT, Copilot, and other coding agents understand the project quickly and avoid common mistakes.

## What This Project Is

NPU MFU Analyzer diagnoses Ascend NPU LLM training bottlenecks from profiling traces. It turns raw profiling outputs into MFU summaries, communication/idle/overlap breakdowns, structured "main contradiction" diagnosis, and prioritized optimization actions.

The package import root is:

```python
npu_mfu_analyzer
```

Do not use the legacy `src.*` import path in new code or docs.

## Common Commands

```bash
pip install -e ".[dev]"

npu-analyzer version
npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/sample_report.md
```

For Zhipu GLM via the Anthropic-compatible Claude backend:

```bash
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_api_key"
npu-analyzer analyze examples/sample_profiling_small -b claude -m GLM-4.7
```

Never write real API keys into files, tests, docs, examples, commits, or CI logs.

## Key Entry Points

| Area | File |
| --- | --- |
| CLI | `src/npu_mfu_analyzer/cli/main.py` |
| Analysis orchestration | `src/npu_mfu_analyzer/agents/orchestrator.py` |
| Profiling loader | `src/npu_mfu_analyzer/data_loader/profiling_loader.py` |
| Streaming trace parser | `src/npu_mfu_analyzer/data_loader/stream_parser.py` |
| Skills v2 engine | `src/npu_mfu_analyzer/skills/v2/` |
| LLM backends | `src/npu_mfu_analyzer/llm/llm_interface.py` |
| Reports | `src/npu_mfu_analyzer/report/` |

## Safe Defaults

- Use `-b mock` for tests and docs unless a real LLM call is explicitly required.
- Use `examples/sample_profiling_small` for smoke tests.
- Keep Host-Device correlation disabled for small docs examples.
- Large `trace_view.json` files are guarded by default; use `--full-host-device-correlation` only when intentionally testing full correlation.

## Development Rules

- Keep public version strings consistent with `pyproject.toml`.
- Prefer structured parser/loader APIs over ad hoc CSV or JSON string parsing.
- Add tests for CLI-visible behavior, parsing fixes, and diagnosis logic changes.
- Preserve graceful degradation when optional profiling tables are missing.
- Treat real profiling data as sensitive; sanitize paths, user names, task names, cluster metadata, and secrets before committing fixtures.

## Useful Test Commands

```bash
python -m py_compile \
  src/npu_mfu_analyzer/cli/main.py \
  src/npu_mfu_analyzer/agents/orchestrator.py \
  src/npu_mfu_analyzer/llm/llm_interface.py

pytest -q tests/unit/test_llm.py tests/unit/test_data_loader.py tests/unit/test_diagnosis_skills.py
```

## Project Links

- English README: `README.md`
- Chinese README: `README.zh-CN.md`
- Sample profiling fixture: `examples/sample_profiling_small`
- Installation: `docs/installation.md`
- Configuration: `docs/configuration.md`
- API reference: `docs/api_reference.md`
