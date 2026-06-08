# 小型 Profiling 样例数据

这个目录包含一份真实昇腾 NPU profiling 的极小脱敏截断子集，用于快速 smoke test、文档示例和 CI 检查。

## 包含内容

- 单 rank 目录：`ascend_sample_rank0`
- Profiling 元信息：`profiler_metadata.json`、`profiler_info_0.json`
- 轻量 CSV 摘要：
  - `ASCEND_PROFILER_OUTPUT/step_trace_time.csv`
  - `ASCEND_PROFILER_OUTPUT/op_statistic.csv`
  - `ASCEND_PROFILER_OUTPUT/kernel_details.csv`

`kernel_details.csv` 只保留原始数据的表头和前 200 行。大型 trace、数据库文件、日志和通信 JSON 都不会提交到仓库里。

## 快速验证

```bash
npu-analyzer info examples/sample_profiling_small
npu-analyzer summary examples/sample_profiling_small --max-steps 10
npu-analyzer analyze examples/sample_profiling_small -b mock --no-host-device-correlation -o /tmp/sample_report.md
```

## 限制

这份样例只用于证明解析器、CLI 和报告链路可用。它不是完整 benchmark，也不应用来判断真实训练任务的最终性能结论。

