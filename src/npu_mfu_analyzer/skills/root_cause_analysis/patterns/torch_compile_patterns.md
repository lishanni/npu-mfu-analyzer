# torch.compile 堆栈模式库

## 编译模式特征函数

### 动态编译相关
- `CompiledFunctionBackward` - 编译函数的反向传播
- `CompiledFunction` - 编译后的函数
- `CUDAGraph` - CUDA Graph 模式
- `TorchDynamo` - Dynamo 编译器
- `AOTAutograd` - 提前编译自动求导

### Dynamo 内部函数
- `torch._dynamo`
- `torch._inductor`
- `_dynamo_eval_frame`
- `dynamo_wrapped`
- `OptimizedModule`

### 编译优化相关
- `compile_inner`
- `wrapped_with_codegen`
- `graph_break`

## 堆栈识别示例

### 完整 torch.compile 堆栈

```
File "torch/_dynamo/eval_frame.py", line 123, in _dynamo_eval_frame
File "torch/_inductor/compile_fx.py", line 456, in compile_fx
File "torch/_inductor/codegen/wrapper.py", line 78, in call
```

### 部分编译堆栈

```
File "torch/_dynamo/eval_frame.py", line 123, in _dynamo_eval_frame
File "model.py", line 45, in forward  # 回退到 eager
```

## 与 eager 模式的区别

| 特征 | torch.compile | eager |
|------|--------------|-------|
| 堆栈深度 | 更深（包含编译层） | 较浅 |
| 函数名 | 包含 Compiled/Dynamo | 直接是 aten:: |
| 调用链 | 有额外的优化层 | 直接调用 |