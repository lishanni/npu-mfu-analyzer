"""
MFU 计算技能

精确计算 Model FLOPS Utilization
"""

from typing import Dict, Any, List
import logging

from ..base_skill import (
    BaseSkill,
    SkillMetadata,
    SkillCategory,
    SkillPriority,
    SkillInput,
    SkillOutput,
    SkillResult,
)

logger = logging.getLogger(__name__)


class CalculateMFUSkill(BaseSkill):
    """
    计算 MFU (Model FLOPS Utilization)
    
    MFU = 实际计算 FLOPS / 理论峰值 FLOPS
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="calculate_mfu",
            display_name="计算 MFU",
            description="计算模型的 FLOPS 利用率，对比实际性能与硬件理论峰值",
            category=SkillCategory.COMPUTE,
            priority=SkillPriority.CRITICAL,
            version="1.0.0",
            inputs=[
                SkillInput(
                    name="model_flops",
                    type="float",
                    required=True,
                    description="模型每次前向+反向传播的总 FLOPS",
                ),
                SkillInput(
                    name="step_time_ms",
                    type="float",
                    required=True,
                    description="单个训练步骤的时间（毫秒）",
                ),
                SkillInput(
                    name="peak_tflops",
                    type="float",
                    required=True,
                    description="硬件理论峰值算力（TFLOPS）",
                ),
                SkillInput(
                    name="num_gpus",
                    type="int",
                    required=False,
                    default=1,
                    description="使用的 GPU/NPU 数量",
                ),
                SkillInput(
                    name="precision",
                    type="str",
                    required=False,
                    default="fp16",
                    description="计算精度 (fp16/bf16/fp32)",
                ),
            ],
            outputs=[
                SkillOutput(name="mfu", type="float", description="MFU 百分比"),
                SkillOutput(name="actual_tflops", type="float", description="实际算力 TFLOPS"),
                SkillOutput(name="theoretical_tflops", type="float", description="理论算力 TFLOPS"),
                SkillOutput(name="efficiency_level", type="str", description="效率等级"),
            ],
            tags=["mfu", "flops", "performance", "compute", "efficiency"],
        )
    
    def execute(
        self,
        model_flops: float,
        step_time_ms: float,
        peak_tflops: float,
        num_gpus: int = 1,
        precision: str = "fp16",
        **kwargs,
    ) -> SkillResult:
        """执行 MFU 计算"""
        
        # 计算实际 TFLOPS
        step_time_s = step_time_ms / 1000.0
        actual_flops_per_second = model_flops / step_time_s
        actual_tflops = actual_flops_per_second / 1e12
        
        # 总理论峰值（考虑多卡）
        theoretical_tflops = peak_tflops * num_gpus
        
        # 计算 MFU
        mfu = (actual_tflops / theoretical_tflops) * 100 if theoretical_tflops > 0 else 0
        
        # 确定效率等级和建议
        suggestions = []
        if mfu >= 50:
            efficiency_level = "优秀"
            suggestions.append("MFU > 50%，性能表现优秀")
        elif mfu >= 35:
            efficiency_level = "良好"
            suggestions.append("MFU 35-50%，性能良好，仍有优化空间")
            suggestions.append("建议检查通信掩盖率和内存带宽利用率")
        elif mfu >= 20:
            efficiency_level = "一般"
            suggestions.append("MFU 20-35%，存在明显瓶颈")
            suggestions.append("建议分析：1) 通信占比 2) Kernel Launch 开销 3) 内存碎片")
        else:
            efficiency_level = "较差"
            suggestions.append("MFU < 20%，性能严重不足")
            suggestions.append("优先排查：1) 数据加载瓶颈 2) Host-Device 同步 3) 算子实现效率")
        
        # 添加精度相关建议
        if precision == "fp32" and mfu < 30:
            suggestions.append("当前使用 FP32 精度，建议尝试混合精度训练提升性能")
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "mfu": round(mfu, 2),
                "actual_tflops": round(actual_tflops, 2),
                "theoretical_tflops": round(theoretical_tflops, 2),
                "efficiency_level": efficiency_level,
                "model_flops": model_flops,
                "step_time_ms": step_time_ms,
                "num_gpus": num_gpus,
                "precision": precision,
            },
            summary=f"MFU = {mfu:.1f}% ({efficiency_level})，实际算力 {actual_tflops:.1f} TFLOPS / 理论 {theoretical_tflops:.1f} TFLOPS",
            suggestions=suggestions,
            confidence=0.95,
        )


class EstimateModelFLOPsSkill(BaseSkill):
    """
    估算模型 FLOPS
    
    基于模型参数估算单次前向+反向传播的 FLOPS
    """
    
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="estimate_model_flops",
            display_name="估算模型 FLOPS",
            description="根据模型参数估算单次训练迭代的 FLOPS",
            category=SkillCategory.COMPUTE,
            priority=SkillPriority.HIGH,
            version="1.0.0",
            inputs=[
                SkillInput(name="num_params", type="float", required=True,
                          description="模型参数量（B）"),
                SkillInput(name="batch_size", type="int", required=True,
                          description="Batch size"),
                SkillInput(name="seq_length", type="int", required=True,
                          description="序列长度"),
                SkillInput(name="num_layers", type="int", required=False, default=0,
                          description="层数（可选，用于更精确估算）"),
                SkillInput(name="hidden_size", type="int", required=False, default=0,
                          description="隐藏层大小（可选）"),
            ],
            outputs=[
                SkillOutput(name="forward_flops", type="float", description="前向 FLOPS"),
                SkillOutput(name="backward_flops", type="float", description="反向 FLOPS"),
                SkillOutput(name="total_flops", type="float", description="总 FLOPS"),
            ],
            tags=["flops", "model", "estimation"],
        )
    
    def execute(
        self,
        num_params: float,
        batch_size: int,
        seq_length: int,
        num_layers: int = 0,
        hidden_size: int = 0,
        **kwargs,
    ) -> SkillResult:
        """估算 FLOPS"""
        
        # 转换参数量到实际数值（B -> 实际）
        params = num_params * 1e9
        
        # 简化估算公式：
        # Forward FLOPS ≈ 2 * params * batch_size * seq_length
        # (每个参数在前向传播中参与 ~2 次乘加操作)
        forward_flops = 2 * params * batch_size * seq_length
        
        # 反向传播约为前向的 2 倍
        backward_flops = 2 * forward_flops
        
        # 总 FLOPS
        total_flops = forward_flops + backward_flops
        
        # 如果提供了更详细的参数，使用更精确的公式
        if num_layers > 0 and hidden_size > 0:
            # Transformer 模型更精确的估算
            # 参考: https://arxiv.org/abs/2001.08361
            attention_flops = 4 * batch_size * seq_length * seq_length * hidden_size
            ffn_flops = 16 * batch_size * seq_length * hidden_size * hidden_size
            layer_flops = attention_flops + ffn_flops
            forward_flops = num_layers * layer_flops
            backward_flops = 2 * forward_flops
            total_flops = forward_flops + backward_flops
        
        return SkillResult(
            skill_name=self.metadata.name,
            success=True,
            data={
                "forward_flops": forward_flops,
                "backward_flops": backward_flops,
                "total_flops": total_flops,
                "total_tflops": total_flops / 1e12,
            },
            summary=f"估算总 FLOPS: {total_flops/1e12:.2f} TFLOPS (前向 {forward_flops/1e12:.2f}T + 反向 {backward_flops/1e12:.2f}T)",
            confidence=0.8 if (num_layers > 0 and hidden_size > 0) else 0.6,
        )
