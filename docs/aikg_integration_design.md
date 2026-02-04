# AIKG 集成设计文档

## 概述

本文档描述如何将 AKG 的 AIKG 模块集成到 npu-mfu-analyzer 的融合机会分析中，实现自动化的融合算子生成。

## 背景

- **AKG (Auto Kernel Generator)**: 华为开源的深度学习算子优化器，基于多面体编译技术
- **AIKG**: AKG 的 AI 驱动内核生成器，使用 LLM 进行多智能体协作，自动生成融合算子
- **npu-mfu-analyzer**: NPU MFU 分析工具，能够检测融合机会

## 集成架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    npu-mfu-analyzer                              │
├─────────────────────────────────────────────────────────────────┤
│  Profiling Data → FusionAnalyzer → FusionOpportunities           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              AIKG Integration Module (新增)                      │
├─────────────────────────────────────────────────────────────────┤
│  • FusionOpportunity → AIKGRequest                              │
│  • AIKG Kernel Generation → FusionKernel                        │
│  • Kernel Verification → Performance Benchmark                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AIKG Service                                  │
├─────────────────────────────────────────────────────────────────┤
│  • LLM-based Code Generation                                     │
│  • Triton-Ascend Kernel Generation                               │
│  • Multi-agent Collaboration                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. AIKGRequestConverter

将 `FusionOpportunity` 转换为 AIKG 可处理的请求格式。

```python
@dataclass
class AIKGRequest:
    """AIKG 生成请求"""
    fusion_name: str           # 融合名称
    operator_sequence: List[str]  # 算子序列
    input_shapes: List[List[int]]  # 输入形状
    output_shapes: List[List[int]] # 输出形状
    data_types: List[str]      # 数据类型
    target_backend: str = "ascend"  # 目标后端
    optimization_level: str = "O2"   # 优化级别
```

### 2. AIKGKernelClient

与 AIKG 服务通信的客户端，支持：
- 同步/异步调用
- 批量生成
- 错误重试

### 3. GeneratedKernel

生成的融合算子表示：

```python
@dataclass
class GeneratedKernel:
    """生成的融合算子"""
    kernel_name: str
    triton_code: str
    ascend_build_script: str
    compilation_status: str  # "success" | "failed" | "pending"
    performance_estimate: Dict[str, float]
```

### 4. KernelValidator

验证生成的融合算子：
- 正确性验证（与原始算子对比）
- 性能验证（实际加速比）
- 内存占用验证

## 工作流程

### Phase 1: 融合机会检测

```
FusionAnalyzer.detect_opportunities()
    ↓
List[FusionOpportunity]
    - opportunity_type: "fuse" | "replace" | "custom"
    - current_ops: 涉及的算子列表
    - estimated_speedup: 估算加速比
```

### Phase 2: AIKG 请求生成

```python
converter = AIKGRequestConverter()
aikg_requests = converter.convert_opportunities(fusion_opportunities)
```

转换规则：
| FusionOpportunity.opportunity_type | AIKG Action |
|-------------------------------------|-------------|
| "replace" (昇腾已有) | 跳过（直接使用原生算子） |
| "fuse" (需要融合) | 生成融合算子 |
| "custom" (自定义) | 生成自定义算子 |

### Phase 3: AIKG 算子生成

```python
client = AIKGKernelClient(aikg_service_url="http://localhost:8080")
generated_kernels = await client.generate_kernels(aikg_requests)
```

### Phase 4: 算子验证与集成

```python
validator = KernelValidator()
validation_results = validator.validate(generated_kernels, reference_data)
```

## 集成点设计

### 在 OperatorAgent 中集成

```python
class OperatorAgent(BaseAgent):
    def __init__(self, llm, config=None):
        super().__init__(...)
        self._fusion_analyzer = FusionAnalyzer()
        self._aikg_converter = AIKGRequestConverter()
        self._aikg_client = AIKGKernelClient()  # 可选

    async def analyze(self, data):
        # 原有分析
        analysis_data = self._prepare_analysis_data(data)
        fusion_opportunities = self._fusion_analyzer.detect_opportunities(...)

        # 新增：AIKG 集成
        if config.get("enable_aikg", False):
            aikg_kernels = await self._generate_fusion_kernels(fusion_opportunities)
            analysis_data.generated_kernels = aikg_kernels

        return result
```

### CLI 命令扩展

```bash
# 新增 AIKG 相关命令
npu-analyzer analyze --aikg /path/to/profiling
npu-analyzer fusion-generate --opportunity-id 123 --output ./kernels/
npu-analyzer kernel-validate --kernel-path ./kernels/fused_matmul.py
```

## 配置管理

```yaml
# config/aikg_config.yaml
aikg:
  enabled: false  # 默认关闭，需要显式启用
  service_url: "http://localhost:8080"
  timeout: 300  # 5分钟超时
  max_concurrent: 3  # 最多并发生成3个算子

  # 生成策略
  strategy:
    min_speedup_threshold: 1.2  # 只生成预期加速比 >= 1.2 的融合
    max_complexity: "高"         # 包含"低"、"中等"、"高"
    skip_native_ops: true       # 跳过昇腾已有算子

  # 输出配置
  output:
    kernel_dir: "./generated_kernels"
    build_script: true          # 生成编译脚本
    benchmark: true             # 生成性能测试代码
```

## 数据流

```
Profiling Data
    ↓
[OperatorAgent.analyze]
    ↓
FusionOpportunities
    ├── Ascend Native Ops → 直接使用
    └── Custom Fusion Ops → [AIKGConverter]
                           ↓
                       AIKG Request
                           ↓
                       [AIKG Service]
                           ↓
                       Generated Kernel
                           ├── Triton Code (.py)
                           ├── Build Script (.sh)
                           └── Benchmark Code (.py)
                           ↓
                       [KernelValidator]
                           ↓
                       Validation Report
                           ↓
                       AnalysisResult (包含生成的算子)
```

## 错误处理

1. **AIKG 服务不可用**: 降级为仅检测融合机会，不生成算子
2. **算子生成失败**: 记录错误，跳过该融合机会，继续处理其他
3. **验证失败**: 标记为"需要人工检查"，不阻止整体流程

## 扩展性

1. **支持多种后端**: 扩展支持 GPU、CPU 等
2. **自定义生成模板**: 允许用户提供算子生成模板
3. **增量生成**: 缓存已生成的算子，避免重复生成

## 实现优先级

| 阶段 | 功能 | 优先级 |
|------|------|--------|
| P0 | AIKGRequestConverter 数据结构 | 高 |
| P0 | 基础 AIKG 客户端框架 | 高 |
| P1 | 与 OperatorAgent 集成 | 高 |
| P1 | 单元测试 | 高 |
| P2 | KernelValidator | 中 |
| P2 | CLI 命令扩展 | 中 |
| P3 | 性能基准测试 | 低 |

## 参考资料

- AKG 仓库: https://atomgit.com/mindspore/akg
- AIKG 文档: https://www.mindspore.cn/docs
- Triton 文档: https://triton-lang.org/
