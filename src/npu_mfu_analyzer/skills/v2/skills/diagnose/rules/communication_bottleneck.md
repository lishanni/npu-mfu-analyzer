---
name: diagnose_communication_bottleneck
type: diagnose
category: communication
priority: P0
tags: [communication, bottleneck, hccl]
dependencies: [compute_bandwidth, compute_overlap]
---

# 通信瓶颈诊断

## 触发条件

```yaml
conditions:
  - type: bandwidth_efficiency
    threshold: 50%
    operator: "<"
  - type: exposed_comm_ratio
    threshold: 30%
    operator: ">"
  - type: slow_rank_detected
    count: 1
```

## 根因描述

通信效率低下，存在明显的通信瓶颈。

## 证据模式

1. 带宽利用率 < 50%
2. 暴露通信时间占比 > 30%
3. 存在慢卡或网络异常
4. HCCL 通信耗时异常

## 诊断步骤

1. 检查 HCCL 配置参数
2. 检查网络拓扑和链路状态
3. 检查是否存在慢卡
4. 检查通信掩盖策略

## 优化建议

### 方案 1: 调整 HCCL 参数

```bash
# 增大通信缓冲区
export HCCL_BUFFSIZE=120
# 启用通信掩盖
export HCCL_INTRA_ROCE_ENABLE=1
```

### 方案 2: 优化通信掩盖

```python
# 启用梯度异步通信
torch.distributed.all_reduce(tensor, async_op=True)
```

### 方案 3: 检查网络拓扑

- 确认 HCCS 链路正常
- 检查 RDMA 配置
- 排查网络拥塞

## 影响范围评估

- **性能影响**: 高 - 可能导致 20-40% 性能下降
- **修复难度**: 中 - 需要系统级调优
- **优先级**: P0 - 应立即处理