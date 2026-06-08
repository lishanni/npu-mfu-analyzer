# Contributing / 贡献指南

Thank you for helping improve NPU MFU Analyzer. Contributions are welcome from AI infrastructure engineers, Ascend NPU users, profiling tool authors, and LLM training practitioners.

感谢你帮助改进 NPU MFU Analyzer。我们欢迎 AI Infra、昇腾 NPU、profiling 工具和大模型训练优化方向的贡献。

## Development Setup

```bash
git clone https://github.com/lishanni/npu-mfu-analyzer.git
cd npu-mfu-analyzer
pip install -e ".[dev]"
```

Quick checks:

```bash
npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/report.md
pytest -q tests/unit/test_llm.py tests/unit/test_data_loader.py tests/unit/test_diagnosis_skills.py
```

## Pull Request Guidelines

- Keep changes focused and explain the profiling scenario they improve.
- Add or update tests for parser, CLI, diagnosis, or report behavior.
- Use `npu_mfu_analyzer.*` imports. Do not add legacy `src.*` imports.
- Prefer `-b mock` in tests and docs unless the change explicitly validates a real LLM backend.
- Document fallback behavior when profiling tables are missing.

## Profiling Data Contributions

Profiling data is often sensitive. Before attaching data to an issue or PR:

- Remove user names, host names, internal paths, job names, and cluster IDs.
- Remove API keys, tokens, command-line secrets, and environment secrets.
- Prefer a tiny reproducer under 1 MB.
- Describe CANN, torch-npu, PyTorch, model family, rank count, and export options.

中文提示：提交 profiling 适配问题时，请尽量提供脱敏后的最小样例和环境信息，不要上传原始内部 trace 或密钥。

