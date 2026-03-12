# mindspeed 堆栈模式库

## MindSpeed/Megatron 特征函数

### 并行层相关
- `ParallelMLP` - 并行 MLP 层
- `ParallelAttention` - 并行注意力层
- `ColumnParallelLinear` - 列并行线性层
- `RowParallelLinear` - 行并行线性层
- `VocabParallelEmbedding` - 词表并行嵌入层

### Transformer 相关
- `TransformerBlock` - Transformer 块
- `transformer_module` - Transformer 模块
- `ParallelTransformerLayer` - 并行 Transformer 层

### Tensor Parallel 相关
- `TensorParallel` - 张量并行
- `SequenceParallel` - 序列并行
- `fused_weight_gradient` - 融合权重梯度

### Pipeline Parallel 相关
- `PipelineParallel` - 流水线并行
- `PipelineStage` - 流水线阶段

## 堆栈识别示例

### MindSpeed 并行 MLP 堆栈

```
File "mindspeed/parallel/mlp.py", line 123, in forward
File "mindspeed/parallel/layers.py", line 456, in ColumnParallelLinear
File "torch/nn/modules/linear.py", line 78, in forward
```

### Megatron Transformer 堆栈

```
File "megatron/model/transformer.py", line 123, in forward
File "megatron/model/layers.py", line 456, in ParallelAttention
```

## 与原生 PyTorch 的区别

| 特征 | MindSpeed/Megatron | 原生 PyTorch |
|------|-------------------|--------------|
| 并行层 | ColumnParallel/RowParallel | Linear |
| 通信 | 内置 all_reduce | 需手动添加 |
| 融合 | 内置融合算子 | 分离算子 |