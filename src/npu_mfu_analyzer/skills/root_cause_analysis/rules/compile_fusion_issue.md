# torch.compile 融合策略问题

## 触发条件

- `stack_pattern`: ["CompiledFunctionBackward", "torch._inductor"]
- `missing_fusion`: ["NPUGroupedMatmul", "FlashAttention"]
- `small_op_increase`: ">50%"
- `operator_types`: ["Add", "zeroslike", "Mul", "Copy"]

## 根因描述

torch.compile 图模式未开启对应的算子融合，导致下发大量小算子。

## 证据模式

- 堆栈中出现 `CompiledFunctionBackward` 但缺少 `NPUGroupedGMM`
- 小算子数量显著增加
- 端到端耗时增加但单个算子耗时无明显变化
- Add、zeroslike 等简单算子数量异常增加

## 优化建议

### 方案 1: 启用 reduce-overhead 模式

```python
model = torch.compile(model, mode="reduce-overhead")
```

### 方案 2: 添加自定义融合规则

```python
@torch.compile
def custom_gmm_forward(x, weight):
    return torch.grouped_matmul(x, weight)
```

### 方案 3: 检查 mindspeed 配置

确保 mindspeed 配置与 torch.compile 兼容：

```python
# 禁用与 torch.compile 冲突的融合
model = torch.compile(
    model,
    disable=not use_custom_fusion
)
```

## 影响范围评估

- **性能影响**: 高 - 可能导致 20-50% 性能下降
- **修复难度**: 中 - 需要调整编译配置
- **优先级**: P0 - 应立即修复

## 典型案例

### 案例描述

对比发现 Device 侧多了很多 Add、zeroslike 算子。

### 分析过程

1. 检查堆栈：发现版本 B 使用 CompiledFunctionBackward
2. 检查融合算子：发现版本 B 缺少 NPUGroupedLinearGMM
3. 统计小算子：版本 B 的小算子数量增加 156%
4. 确定根因：torch.compile 未启用 GroupedMatmul 融合

### 修复效果

启用 reduce-overhead 模式后，小算子数量减少 90%，端到端性能恢复。
