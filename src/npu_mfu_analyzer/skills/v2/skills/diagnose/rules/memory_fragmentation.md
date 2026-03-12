---
name: diagnose_memory_fragmentation
type: diagnose
category: memory
priority: P1
tags: [memory, fragmentation, oom]
dependencies: []
---

# 内存碎片化诊断

## 触发条件

```yaml
conditions:
  - type: memory_fragmentation_ratio
    threshold: 30%
    operator: ">"
  - type: oom_count
    count: 1
  - type: memory_allocation_failures
    count: 10
```

## 根因描述

内存碎片化严重，导致内存利用率低下或 OOM。

## 证据模式

1. 内存碎片率 > 30%
2. 频繁的小内存分配/释放
3. OOM 错误日志
4. 内存分配失败记录

## 诊断步骤

1. 分析内存分配模式
2. 检查是否存在内存泄漏
3. 检查内存复用策略

## 优化建议

### 方案 1: 启用内存池

```python
# 使用 PyTorch 内存池
import torch
torch.cuda.set_per_process_memory_fraction(0.8)
```

### 方案 2: 优化内存分配策略

```python
# 预分配内存
torch.empty((batch_size, hidden_dim), device='npu')
```

### 方案 3: 检查内存泄漏

```python
# 使用内存分析工具
from torch_npu.npu.amp import grad_scaler
# 确保及时释放不再使用的张量
del unused_tensor
torch.npu.empty_cache()
```

## 影响范围评估

- **性能影响**: 中高 - 可能导致 OOM 或性能下降
- **修复难度**: 中 - 需要代码调整
- **优先级**: P1 - 建议尽快处理