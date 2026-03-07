# Root Cause Analysis Skill

## 概述

根因分析技能用于自动识别 Profiling 对比分析中的性能问题根因。

## 能力

- 分析 torch.compile 融合策略差异
- 识别 eager/compile 模式切换
- 检测通信瓶颈
- 发现内存碎片化问题
- 识别分布式训练问题

## 触发条件

当执行 `compare` 命令时，如果检测到显著性能差异，自动触发根因分析。

## 输入

- `chains_a`: 版本 A 的 Host-Device 调用链
- `chains_b`: 版本 B 的 Host-Device 调用链
- `diff_result`: ProfilingDiff 对比结果

## 输出

- `root_cause_findings`: 根因发现列表
- `optimization_suggestions`: 优化建议

## 规则文件

参见 `rules/` 目录下的各规则文件。

## 模式库

参见 `patterns/` 目录下的堆栈模式库。
