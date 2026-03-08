# 通信瓶颈问题

## 触发条件

- `comm_ratio_increase`: ">10%" - 通信占比增加超过 10%
- `comm_time_increase`: ">20%" - 通信时间增加超过 20%
- `overlap_ratio_decrease`: ">15%" - 通信掩盖率下降超过 15%
- `slow_links`: ">0" - 存在慢链路

## 根因描述

分布式训练中的通信瓶颈导致性能下降，可能原因包括：
- 通信-计算重叠效率下降
- 存在慢链路（带宽不足或丢包）
- 集合通信模式变化

## 证据模式

- 通信时间占比显著增加
- 未掩盖通信时间增加
- 存在带宽利用率低于阈值的链路
- all_reduce/all_gather/reduce_scatter 时间变化

## 优化建议

### 方案 1: 优化通信掩盖

```python
# 使用异步通信
with torch.no_grad():
    handle = dist.all_reduce(tensor, async_op=True)

# 重叠计算
output = model(input)

# 等待通信完成
handle.wait()
```

### 方案 2: 检查网络拓扑

```python
# 检查 HCCS/RDMA 配置
# 确保拓扑匹配通信模式
```

### 方案 3: 调整集合通信算法

```python
# 根据模型大小选择合适的算法
# 小消息: Ring AllReduce
# 大消息: Hierarchical AllReduce
```

## 影响范围评估

- **性能影响**: 高 - 可能导致 10-50% 性能下降
- **修复难度**: 中 - 需要分析具体原因
- **优先级**: P0 - 对分布式训练影响大

## 相关指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 通信占比 | <30% | 理想值 |
| 通信掩盖率 | >50% | 理想值 |
| 慢链路数 | 0 | 目标值 |
| 带宽利用率 | >80% | 理想值 |
