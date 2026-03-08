---
name: diagnose_operator_efficiency
type: diagnose
category: compute
priority: P1
tags: [operator, efficiency, kernel]
dependencies: [compute_mfu]
---

# 算子效率诊断

## 触发条件

```yaml
conditions:
  - type: mfu_value
    threshold: 40%
    operator: "<"
  - type: hot_operator_count
    count: 5
  - type: kernel_launch_overhead
    threshold: 10%
    operator: ">"
```

## 根因描述

算子执行效率低下，可能存在算子性能问题或调度开销过大。

## 证据模式

1. MFU < 40%
2. 存在耗时的热点算子
3. Kernel Launch 开销占比过高
4. 小算子数量过多

## 诊断步骤

1. 识别 Top 热点算子
2. 分析算子耗时分布
3. 检查算子融合情况
4. 分析 Kernel Launch 开销

## 优化建议

### 方案 1: 算子融合

```python
# 使用 torch.compile 启用融合
model = torch.compile(model, mode="reduce-overhead")
```

### 方案 2: 使用融合算子

```python
# 使用 FlashAttention 替代标准 Attention
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v)
```

### 方案 3: 减少 Kernel Launch 开销

```python
# 使用 CUDAGraph 固化计算图
from torch_npu import npu_graph
model = npu_graph.capture(model)
```

## 影响范围评估

- **性能影响**: 中高 - 可能导致 10-30% 性能下降
- **修复难度**: 中 - 需要算子级优化
- **优先级**: P1 - 建议尽快处理