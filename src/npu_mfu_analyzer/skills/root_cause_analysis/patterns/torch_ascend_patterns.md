# torch-ascend 堆栈模式库

## Torch-Ascend/CANN 特征函数

### NPU 算子相关
- `torch_npu` - NPU 扩展模块
- `aten_npu` - NPU 算子实现
- `aclnn` - CANN 算子库接口
- `npu::` - NPU 命名空间

### CANN 相关
- `AscendCL` - Ascend 计算库
- `aclOp` - ACL 算子
- `OpRunner` - 算子运行器
- `AclOpExecutor` - ACL 算子执行器

### NPU 融合算子
- `NPUGroupedLinearGMM` - 分组线性 GMM
- `NPUGroupedMatmul` - 分组矩阵乘
- `npu_scaled_masked_softmax` - NPU 缩放掩码 softmax
- `npu_rotary_mul` - NPU 旋转乘法
- `npu_fusion` - NPU 融合算子

## 堆栈识别示例

### NPU 算子调用堆栈

```
File "torch_npu/npu/ops.py", line 123, in npu_scaled_masked_softmax
File "torch_npu/npu/_C.py", line 456, in _npu_kernel
File "aclnn_scaled_masked_softmax", line 78
```

### CANN 执行堆栈

```
File "torch_npu/csrc/aten/AclOpExecutor.cpp", line 123
File "torch_npu/csrc/aten/OpRunner.cpp", line 456
File "aclnnGroupedMatmul", line 78
```

## 与 CUDA 的区别

| 特征 | Torch-Ascend | CUDA |
|------|-------------|------|
| 算子库 | CANN/aclnn | cuBLAS/cuDNN |
| 融合算子 | NPUGroupedMatmul | cuBLASLt |
| 后端 | torch_npu | torch.cuda |