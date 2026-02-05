"""
算子融合规则库

定义融合规则、模式和昇腾已有的融合算子映射。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import re


class FusionCategory(Enum):
    """融合类别"""
    MATMUL_FUSION = "matmul_fusion"          # MatMul相关融合
    ATTENTION_FUSION = "attention_fusion"    # Attention相关融合
    NORM_FUSION = "norm_fusion"              # 归一化相关融合
    ELEMENT_FUSION = "element_fusion"        # 逐元素操作融合
    COMM_FUSION = "comm_fusion"              # 通信融合
    REDUCTION_FUSION = "reduction_fusion"    # Reduction操作融合


@dataclass
class FusionPattern:
    """融合模式定义"""
    name: str                    # 模式名称
    category: FusionCategory
    operator_patterns: List[str] # 算子名称匹配模式列表（正则）
    memory_saving: float         # 内存节省百分比（估算 0-1）
    speedup: float                # 性能提升倍数（估算）
    triton_complexity: str       # Triton 实现复杂度
    description: str             # 描述
    example_code: str = ""       # 示例代码片段
    ascend_op: Optional[str] = None  # 对应的昇腾融合算子（如果存在）


@dataclass
class FusionOpportunity:
    """融合机会"""
    opportunity_type: str  # "replace" | "fuse" | "custom"
    name: str
    description: str
    current_ops: List[Dict[str, Any]]
    estimated_speedup: float  # 算子级别加速比
    end_to_end_speedup: float = 1.0  # 端到端训练加速比（默认 1.0）
    time_proportion: float = 0.0  # 涉及算子占总耗时比例（默认 0）
    memory_saving: float = 0.0
    implementation: str = ""
    complexity: str = "中等"  # "低" | "中等" | "高"
    triton_example: str = ""
    ascend_op: Optional[str] = None
    total_op_duration_us: float = 0  # 涉及算子总耗时（微秒）

    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        # 判断端到端加速级别
        if self.end_to_end_speedup >= 1.1:
            speedup_level = "高"
        elif self.end_to_end_speedup >= 1.03:
            speedup_level = "中"
        else:
            speedup_level = "低"

        lines = [
            f"- **{self.name}** ({self.opportunity_type}) [{speedup_level}优先级]",
            f"  - 描述: {self.description}",
            f"  - 算子级别加速: {self.estimated_speedup:.1f}x",
            f"  - **端到端训练加速: {self.end_to_end_speedup:.1%}** (耗时占比 {self.time_proportion:.1%})",
            f"  - 内存节省: {self.memory_saving*100:.0f}%",
            f"  - 复杂度: {self.complexity}",
            f"  - 实现方式: {self.implementation}",
        ]
        if self.ascend_op:
            lines.append(f"  - 昇腾算子: {self.ascend_op}")
        if self.triton_example:
            lines.append(f"  - Triton 示例: 已提供代码片段")
        return "\n".join(lines)


# =============================================================================
# 昇腾已有融合算子映射表
# =============================================================================
ASCEND_FUSED_OPERATORS: Dict[str, Dict[str, Any]] = {
    # Attention 相关
    "aclnnFlashAttentionScore": {
        "type": "attention",
        "description": "Flash Attention 分数计算融合（支持 FP16/BF16）",
        "performance": "5x+ 性能提升",
        "replaces": ["Attention", "Softmax", "MatMul"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedInferAttentionScoreV2": {
        "type": "attention",
        "description": "融合推理注意力分数 V2（推理优化）",
        "performance": "推理 3-4x 提升",
        "replaces": ["Attention", "Mask", "Softmax"],
        "complexity": "低（直接替换）",
    },
    "aclnnFlashAttentionV2": {
        "type": "attention",
        "description": "Flash Attention V2（支持更大序列长度）",
        "performance": "长序列优化",
        "replaces": ["Attention", "Softmax"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedMultiHeadAttention": {
        "type": "attention",
        "description": "融合多头注意力",
        "performance": "2-3x 提升",
        "replaces": ["QKV Projection", "Attention", "Output Projection"],
        "complexity": "低（直接替换）",
    },

    # Norm 相关
    "aclnnAddRmsNorm": {
        "type": "norm",
        "description": "RMSNorm + Add 残差融合",
        "performance": "20%+ 提升",
        "replaces": ["Add", "RMSNorm"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedLayerNorm": {
        "type": "norm",
        "description": "LayerNorm 融合算子",
        "performance": "15-25% 提升",
        "replaces": ["LayerNorm"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedRMSNorm": {
        "type": "norm",
        "description": "RMSNorm 融合算子",
        "performance": "15-25% 提升",
        "replaces": ["RMSNorm"],
        "complexity": "低（直接替换）",
    },

    # MatMul 相关
    "aclnnGroupedMatmulV4": {
        "type": "matmul",
        "description": "MoE 专家混合矩阵乘法融合",
        "performance": "MoE 优化",
        "replaces": ["GroupedMatMul"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedMatMulBias": {
        "type": "matmul",
        "description": "MatMul + Bias 融合",
        "performance": "10-20% 提升",
        "replaces": ["MatMul", "BiasAdd"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedMatMulBiasAct": {
        "type": "matmul",
        "description": "MatMul + Bias + Activation 融合",
        "performance": "20-30% 提升",
        "replaces": ["MatMul", "BiasAdd", "GELU", "ReLU", "SiLU"],
        "complexity": "低（直接替换）",
    },

    # 逐元素操作
    "aclnnFusedScaleSoftmax": {
        "type": "element",
        "description": "Scale + Softmax 融合",
        "performance": "15% 提升",
        "replaces": ["Mul", "Softmax"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedDropoutAdd": {
        "type": "element",
        "description": "Dropout + Add 融合",
        "performance": "10% 提升",
        "replaces": ["Dropout", "Add"],
        "complexity": "低（直接替换）",
    },

    # QKV Projection 融合
    "aclnnFusedQKVProjection": {
        "type": "attention",
        "description": "QKV 融合投影",
        "performance": "2x 提升",
        "replaces": ["MatMul_Q", "MatMul_K", "MatMul_V"],
        "complexity": "低（直接替换）",
    },

    # GPT/Gemma 专用融合
    "aclnnGemmaFusedRMSNorm": {
        "type": "norm",
        "description": "Gemma RMSNorm 融合",
        "performance": "Gemma 模型优化",
        "replaces": ["RMSNorm"],
        "complexity": "低（直接替换）",
    },
    "aclnnFusedGeGLU": {
        "type": "element",
        "description": "GeGLU 激活函数融合",
        "performance": "GeGLU 优化",
        "replaces": ["MatMul", "Gate", "Mul"],
        "complexity": "低（直接替换）",
    },
}


# =============================================================================
# 通用融合模式库（用于自定义融合建议）
# =============================================================================
FUSION_PATTERNS: List[FusionPattern] = [
    # MatMul 相关融合
    FusionPattern(
        name="MatMul+Bias+GELU/ReLU/SiLU",
        category=FusionCategory.MATMUL_FUSION,
        operator_patterns=[
            r"MatMul.*",
            r"Mul.*Bias.*|BiasAdd.*",
            r"GELU|ReLU|SiLU|Sigmoid"
        ],
        memory_saving=0.4,
        speedup=1.3,
        triton_complexity="中等",
        description="将矩阵乘法、偏置加法和激活函数融合为一个 Kernel",
        ascend_op="aclnnFusedMatMulBiasAct",
        example_code="""
@triton.jit
def fused_matmul_bias_gelu(
    A, B, bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Fused MatMul + Bias + GELU
    pid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # ... 省略详细实现
""",
    ),

    FusionPattern(
        name="QKV Projection Merge",
        category=FusionCategory.ATTENTION_FUSION,
        operator_patterns=[
            r"MatMul.*Q|MatMul.*query",
            r"MatMul.*K|MatMul.*key",
            r"MatMul.*V|MatMul.*value"
        ],
        memory_saving=0.35,
        speedup=1.5,
        triton_complexity="中等",
        description="将 Q、K、V 三个独立投影合并为一个大矩阵乘法",
        ascend_op="aclnnFusedQKVProjection",
        example_code="""
# 合并前: 3 个独立 MatMul
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# 合并后: 1 个 MatMul
QKV = x @ W_qkv  # W_qkv = concat(W_q, W_k, W_v)
Q, K, V = QKV.chunk(3, dim=-1)
""",
    ),

    # Norm 相关融合
    FusionPattern(
        name="LayerNorm/ RMSNorm + Residual Add",
        category=FusionCategory.NORM_FUSION,
        operator_patterns=[
            r"LayerNorm|RMSNorm",
            r"Add.*residual|Residual.*Add"
        ],
        memory_saving=0.3,
        speedup=1.2,
        triton_complexity="简单",
        description="将归一化和残差连接融合，减少中间结果存储",
        ascend_op="aclnnAddRmsNorm",
        example_code="""
@triton.jit
def fused_rmsnorm_residual(
    x, residual, weight, bias, y,
    stride, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Fused RMSNorm + Residual Add
    # ... 省略详细实现
""",
    ),

    FusionPattern(
        name="LayerNorm + Reshape + Transpose",
        category=FusionCategory.NORM_FUSION,
        operator_patterns=[
            r"LayerNorm",
            r"Reshape",
            r"Transpose"
        ],
        memory_saving=0.25,
        speedup=1.15,
        triton_complexity="中等",
        description="融合归一化后的形状变换操作",
        ascend_op=None,
    ),

    # Attention 相关融合
    FusionPattern(
        name="RotaryEmbedding + QKV Projection",
        category=FusionCategory.ATTENTION_FUSION,
        operator_patterns=[
            r"ApplyRotaryPosEmb|RotaryEmb",
            r"MatMul.*Q"
        ],
        memory_saving=0.2,
        speedup=1.2,
        triton_complexity="复杂",
        description="将旋转位置编码与 QKV 投影融合",
        ascend_op=None,
    ),

    FusionPattern(
        name="FlashAttention Replacement",
        category=FusionCategory.ATTENTION_FUSION,
        operator_patterns=[
            r"MatMul.*QK",
            r"Softmax",
            r"MatMul.*Attn"
        ],
        memory_saving=0.6,
        speedup=5.0,
        triton_complexity="复杂",
        description="用 FlashAttention 替换标准 Attention 实现",
        ascend_op="aclnnFlashAttentionScore",
        example_code="""
# 标准 Attention (需要优化)
attn_scores = Q @ K.T / sqrt(d)
attn_weights = softmax(attn_scores, mask)
output = attn_weights @ V

# FlashAttention (直接替换)
output = flash_attention(Q, K, V, causal=True)
""",
    ),

    # 逐元素操作融合
    FusionPattern(
        name="Dropout + Add + Multiply",
        category=FusionCategory.ELEMENT_FUSION,
        operator_patterns=[
            r"Dropout",
            r"Add",
            r"Mul|Scale"
        ],
        memory_saving=0.2,
        speedup=1.1,
        triton_complexity="简单",
        description="融合 Dropout、加法和乘法等逐元素操作",
        ascend_op="aclnnFusedDropoutAdd",
    ),

    FusionPattern(
        name="Activation + Mul (SiLU/GLU variants)",
        category=FusionCategory.ELEMENT_FUSION,
        operator_patterns=[
            r"SiLU|Sigmoid",
            r"Mul"
        ],
        memory_saving=0.15,
        speedup=1.1,
        triton_complexity="简单",
        description="融合激活函数和乘法（如 SiLU(x) * x）",
        ascend_op=None,
        example_code="""
@triton.jit
def fused_silu_mul(x, y, BLOCK_SIZE: tl.constexpr):
    # Fused SiLU(x) * x (SwiGLU variant)
    # ... 省略详细实现
""",
    ),

    # Reduction 融合
    FusionPattern(
        name="Softmax + MatMul (Attention output)",
        category=FusionCategory.ATTENTION_FUSION,
        operator_patterns=[
            r"Softmax",
            r"MatMul.*V"
        ],
        memory_saving=0.3,
        speedup=1.2,
        triton_complexity="中等",
        description="融合 Attention 中的 Softmax 和输出 MatMul",
        ascend_op="aclnnFlashAttentionScore",
    ),

    # MoE 相关融合
    FusionPattern(
        name="MoE Routing + Expert MatMul",
        category=FusionCategory.MATMUL_FUSION,
        operator_patterns=[
            r"TopK|Routing",
            r"MatMul.*expert"
        ],
        memory_saving=0.4,
        speedup=1.4,
        triton_complexity="复杂",
        description="融合 MoE 路由选择和专家计算",
        ascend_op="aclnnGroupedMatmulV4",
    ),

    # Casting 融合
    FusionPattern(
        name="Cast + MatMul (Type conversion fusion)",
        category=FusionCategory.ELEMENT_FUSION,
        operator_patterns=[
            r"Cast",
            r"MatMul"
        ],
        memory_saving=0.1,
        speedup=1.05,
        triton_complexity="简单",
        description="将数据类型转换融合到 MatMul 中",
        ascend_op=None,
    ),

    # 简单逐元素操作融合
    FusionPattern(
        name="Add + Mul (Element-wise fusion)",
        category=FusionCategory.ELEMENT_FUSION,
        operator_patterns=[
            r"Add",
            r"Mul"
        ],
        memory_saving=0.15,
        speedup=1.1,
        triton_complexity="简单",
        description="融合加法和乘法操作",
        ascend_op=None,
        example_code="""
@triton.jit
def fused_add_mul(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Fused: (x + y) * z
    result = (x + y) * z

    tl.store(output_ptr + offsets, result, mask=mask)
""",
    ),

    # Cast + 具体算子融合
    FusionPattern(
        name="Cast to FP16 + MatMulV2",
        category=FusionCategory.ELEMENT_FUSION,
        operator_patterns=[
            r"Cast.*",
            r"MatMulV2"
        ],
        memory_saving=0.2,
        speedup=1.15,
        triton_complexity="简单",
        description="将 FP32->FP16 转换融合到 MatMulV2 中",
        ascend_op=None,
    ),
]


def get_ascend_fused_op_names() -> List[str]:
    """获取所有昇腾融合算子名称列表"""
    return list(ASCEND_FUSED_OPERATORS.keys())


def find_ascend_fused_op(patterns: List[str]) -> Optional[str]:
    """
    根据算子模式查找对应的昇腾融合算子

    Args:
        patterns: 算子名称模式列表

    Returns:
        匹配的昇腾融合算子名称，如果没有则返回 None
    """
    pattern_str = "|".join(patterns)

    for ascend_op, info in ASCEND_FUSED_OPERATORS.items():
        replaces = info.get("replaces", [])
        for replace_op in replaces:
            if re.search(pattern_str, replace_op, re.IGNORECASE):
                return ascend_op

    return None


def get_fusion_pattern_by_name(name: str) -> Optional[FusionPattern]:
    """根据名称获取融合模式"""
    for pattern in FUSION_PATTERNS:
        if pattern.name == name:
            return pattern
    return None


def get_fusion_patterns_by_category(category: FusionCategory) -> List[FusionPattern]:
    """获取指定类别的融合模式"""
    return [p for p in FUSION_PATTERNS if p.category == category]
