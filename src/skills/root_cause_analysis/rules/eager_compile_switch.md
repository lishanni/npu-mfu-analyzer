# Eager/Compile 模式切换问题

## 触发条件

- `source_change`: eager → torch_compile 或 torch_compile → eager
- `operator_overlap`: "<80%" - 两个版本的算子重叠度低
- `execution_mode_diff`: True

## 根因描述

两个 Profiling 版本使用了不同的执行模式（eager vs torch.compile），导致性能表现差异。

## 证据模式

- 版本 A 堆栈显示 `aten::` 开头的原生算子
- 版本 B 堆栈显示 `CompiledFunction` 或 `torch._dynamo`
- 或者相反
- 两个版本的算子数量差异大
- 算子名称出现显著变化（融合 vs 分解）

## 优化建议

### 方案 1: 统一执行模式

确保对比的两个版本使用相同的执行模式：

```python
# 方案 A: 都使用 eager 模式
model = model  # 不编译

# 方案 B: 都使用 torch.compile
model = torch.compile(model)
```

### 方案 2: 分离对比维度

如果要对比 eager vs compile 的性能差异：

1. 先确保模型结构相同
2. 使用相同的数据和配置
3. 分别记录两种模式的 Profiling
4. 单独分析编译开销和收益

## 影响范围评估

- **性能影响**: 高 - eager vs compile 可能差异 30-100%
- **修复难度**: 低 - 只需统一配置
- **优先级**: P0 - 影响对比结论的有效性

## 检测方法

```python
# 检测 eager 模式
def is_eager_mode(stack):
    return "aten::" in stack and "CompiledFunction" not in stack

# 检测 compile 模式
def is_compile_mode(stack):
    return any(p in stack for p in ["CompiledFunction", "torch._dynamo", "torch._inductor"])
```