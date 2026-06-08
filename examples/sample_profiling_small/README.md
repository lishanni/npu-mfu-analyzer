# Small Sample Profiling Data

This directory contains a tiny, sanitized subset of a real Ascend NPU profiling run. It is intended for quick smoke tests, documentation examples, and CI checks.

## What is included

- One rank directory: `ascend_sample_rank0`
- Profiling metadata: `profiler_metadata.json`, `profiler_info_0.json`
- Lightweight CSV summaries:
  - `ASCEND_PROFILER_OUTPUT/step_trace_time.csv`
  - `ASCEND_PROFILER_OUTPUT/op_statistic.csv`
  - `ASCEND_PROFILER_OUTPUT/kernel_details.csv`

`kernel_details.csv` keeps only the header and the first 200 data rows from the original run. Large trace files, database files, logs, and communication JSON files are intentionally omitted.

## Quick checks

```bash
npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/sample_report.md
```

## Limitations

This sample is designed to prove that the parser, CLI, and report path work. It is not a complete benchmark and should not be used to draw final conclusions about a real training job.

