# Copilot Instructions for NPU MFU Analyzer

NPU MFU Analyzer is a Python package for diagnosing Ascend NPU LLM training bottlenecks from profiling traces.

Use these rules when generating code:

- Use `npu_mfu_analyzer.*` imports. Do not use legacy `src.*` imports.
- Keep CLI-visible behavior covered by tests or sample smoke commands.
- Use `examples/sample_profiling_small` for lightweight examples.
- Use `mock` backend for tests unless explicitly validating real LLM behavior.
- Never add real API keys, tokens, host names, user names, internal paths, or proprietary raw traces.
- Preserve graceful fallback when optional profiling tables are missing.
- Large Host-Device correlation should respect the default trace size guardrail.

Key files:

- CLI: `src/npu_mfu_analyzer/cli/main.py`
- Orchestrator: `src/npu_mfu_analyzer/agents/orchestrator.py`
- Loader: `src/npu_mfu_analyzer/data_loader/profiling_loader.py`
- Stream parser: `src/npu_mfu_analyzer/data_loader/stream_parser.py`
- Skills v2: `src/npu_mfu_analyzer/skills/v2/`
- LLM backends: `src/npu_mfu_analyzer/llm/llm_interface.py`

Useful validation:

```bash
npu-analyzer version
npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/report.md
pytest -q tests/unit/test_llm.py tests/unit/test_data_loader.py tests/unit/test_diagnosis_skills.py
```

