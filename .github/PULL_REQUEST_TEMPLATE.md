## Summary

Describe the change and the profiling scenario it improves.

## Validation

- [ ] `pytest -q tests/unit/test_llm.py tests/unit/test_data_loader.py tests/unit/test_diagnosis_skills.py`
- [ ] `npu-analyzer info examples/sample_profiling_small`
- [ ] `npu-analyzer summary examples/sample_profiling_small --max-steps 10`
- [ ] `npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/report.md`

## Data Safety

- [ ] No API keys, tokens, or secrets are included.
- [ ] Profiling data is sanitized or generated.
- [ ] No internal user names, host names, paths, or job names are exposed.

