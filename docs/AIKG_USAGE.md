# npu-mfu-analyzer AIKG 融合算子生成使用指南

## 概述

`npu-mfu-analyzer` 是昇腾 NPU 大模型训练性能分析工具，支持自动化的融合算子生成工作流。通过 `generate` 命令，您可以：

1. 分析 Profiling 数据
2. 自动检测融合机会
3. 调用 AIKG 生成 Triton-Ascend 代码
4. 生成编译脚本和性能测试代码

## 快速开始

### 基本用法

```bash
# 使用默认配置生成融合算子
npu-analyzer generate /path/to/profiling

# 指定输出目录
npu-analyzer generate /path/to/profiling -o ./my_kernels
```

### 使用 Claude API (GLM-4.7)

```bash
# 设置 Anthropic 兼容 API
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"
export ANTHROPIC_API_KEY="your_api_key"

# 使用 Claude 后端生成
npu-analyzer generate /path/to/profiling -b claude -m GLM-4.7
```

## 命令参数

### 必需参数

| 参数 | 说明 |
|------|------|
| `PROFILING_PATH` | Profiling 数据目录路径 |

### 可选参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `./generated_kernels` | 融合算子输出目录 |
| `--backend` | `-b` | `claude` | LLM 后端 (claude/openai/ollama/deepseek/mock) |
| `--model` | `-m` | `GLM-4.7` | LLM 模型名称 |
| `--min-speedup` | - | `1.05` | 最小加速比阈值 |
| `--complexity` | - | `高` | 最大实现复杂度 (低/中等/高) |
| `--skip-native` | - | `True` | 跳过昇腾已有融合算子 |
| `--timeout` | - | `300` | 单个算子生成超时时间（秒） |
| `--max-concurrent` | - | `3` | 最大并发生成数 |
| `--verbose` | `-v` | - | 详细输出 |

## 使用示例

### 示例 1: 生成所有简单的融合算子

```bash
npu-analyzer generate /path/to/profiling \
    --complexity 低 \
    --min-speedup 1.01 \
    --output ./simple_kernels
```

### 示例 2: 不跳过昇腾已有算子（用于验证）

```bash
npu-analyzer generate /path/to/profiling \
    --skip-native false \
    --output ./all_kernels
```

### 示例 3: 使用 Mock 后端测试（不消耗 API）

```bash
npu-analyzer generate /path/to/profiling \
    --backend mock \
    --min-speedup 1.0
```

### 示例 4: 指定并发数和超时

```bash
npu-analyzer generate /path/to/profiling \
    --max-concurrent 1 \
    --timeout 600
```

## 输出文件

运行成功后，会在输出目录生成以下文件：

```
generated_kernels/
├── <fusion_name>.py          # Triton-Ascend 融合算子代码
├── <fusion_name>.sh          # 编译脚本
└── <fusion_name>_bench.py    # 性能测试代码
```

### 文件说明

1. **Triton 代码 (.py)**:
   - 使用 `@triton.jit` 装饰的融合内核
   - 包含前端接口函数
   - 支持自动调优 (`@triton.autotune`)

2. **编译脚本 (.sh)**:
   - 检查 Triton 环境
   - 编译融合算子

3. **性能测试 (_bench.py)**:
   - 基准测试模板
   - 性能对比代码

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Profiling 数据                           │
│                  (msprof, PyTorch Profiler)                 │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   步骤 1: 数据加载                           │
│              检测框架、加载算子信息                          │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   步骤 2: 融合机会检测                       │
│       分析算子序列、匹配融合模式、计算收益                  │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   步骤 3: AIKG 代码生成                     │
│   构建 DSL → LLM API → 生成 Triton-Ascend 代码              │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   步骤 4: 文件保存                           │
│        保存 .py, .sh, _bench.py 到输出目录                 │
└─────────────────────────────────────────────────────────────┘
```

## 支持的融合模式

工具会自动检测以下融合模式：

### 简单融合
- Add + Mul (逐元素操作融合)
- Cast + MatMul (类型转换融合)

### 中等复杂度
- MatMul + Bias + Activation
- LayerNorm/RMSNorm + Residual
- Dropout + Add

### 复杂融合
- FlashAttention (Attention 融合)
- QKV Projection (投影合并)
- MoE 融合 (专家混合)

## 配置 API Key

### Claude API

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

### OpenAI API

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### DeepSeek API

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## 常见问题

### Q1: 没有生成任何融合算子？

可能的原因：
1. `--min-speedup` 设置过高，尝试降低到 `1.01`
2. `--skip-native` 为 True，所有融合机会都是昇腾已有算子
3. `--complexity` 设置过低，尝试设置为 `高`

解决方案：
```bash
npu-analyzer generate /path/to/profiling \
    --min-speedup 1.01 \
    --skip-native false \
    --complexity 高
```

### Q2: 如何只生成简单的融合算子？

```bash
npu-analyzer generate /path/to/profiling \
    --complexity 低 \
    --min-speedup 1.05
```

### Q3: 生成的代码如何使用？

```bash
# 1. 查看生成的代码
cat generated_kernels/<fusion_name>.py

# 2. 编译（如果需要）
bash generated_kernels/<fusion_name>.sh

# 3. 运行性能测试
python generated_kernels/<fusion_name>_bench.py
```

### Q4: 支持哪些 LLM 后端？

- `claude`: Claude API (推荐，支持 GLM-4.7)
- `openai`: OpenAI API (GPT-4, GPT-3.5)
- `ollama`: 本地 Ollama (Llama, Qwen, DeepSeek)
- `deepseek`: DeepSeek API
- `mock`: 测试用，返回模拟代码

### Q5: 如何提高生成质量？

1. 使用更强大的模型（如 GLM-4.7, GPT-4）
2. 在 Profiling 提示中提供更多上下文
3. 调整 `--min-speedup` 和 `--complexity` 参数

## 完整示例

```bash
# 1. 设置 API Key
export ANTHROPIC_API_KEY="your_api_key"

# 2. 运行完整工作流
npu-analyzer generate /path/to/msprof_data \
    --backend claude \
    --model GLM-4.7 \
    --output ./my_fusion_kernels \
    --min-speedup 1.05 \
    --complexity 中等 \
    --max-concurrent 2

# 3. 查看生成的文件
ls -la ./my_fusion_kernels/

# 4. 查看某个融合算子的代码
cat ./my_fusion_kernels/add_mul_fusion.py
```

## 技术支持

如遇问题，请检查：
1. Profiling 数据路径是否正确
2. API Key 是否正确设置
3. 网络连接是否正常
4. 日志输出（使用 `--verbose` 参数）

更多信息请参考：https://github.com/your-repo/npu-mfu-analyzer
