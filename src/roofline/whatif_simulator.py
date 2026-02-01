"""
What-if Simulator 假设分析器

预测不同配置下的性能变化
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """假设场景类型"""
    PARALLEL_CHANGE = "parallel_change"       # 并行度变化
    HARDWARE_UPGRADE = "hardware_upgrade"     # 硬件升级
    PRECISION_CHANGE = "precision_change"     # 精度变化
    BATCH_SIZE_CHANGE = "batch_size_change"   # Batch Size 变化
    OPTIMIZATION = "optimization"             # 优化措施


@dataclass
class CurrentState:
    """当前状态"""
    # 硬件
    hardware_name: str = "Atlas A2 (280T)"
    num_devices: int = 8
    peak_tflops_per_device: float = 280  # FP16
    hbm_bandwidth_gbps: float = 1500
    hccs_bandwidth_gbps: float = 56
    rdma_bandwidth_gbps: float = 25
    
    # 并行配置
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 8
    
    # 模型参数
    model_params_b: float = 7.0  # 参数量（B）
    batch_size: int = 8
    seq_length: int = 4096
    
    # 性能指标
    step_time_ms: float = 1000
    mfu_percent: float = 40
    overlap_ratio: float = 0.7
    comm_time_ratio: float = 0.2
    
    # 计算精度
    precision: str = "bf16"


@dataclass
class WhatIfScenario:
    """假设场景"""
    name: str
    scenario_type: ScenarioType
    description: str
    changes: Dict[str, Any]  # 参数变化
    
    # 预测结果
    predicted_step_time_ms: float = 0
    predicted_mfu_percent: float = 0
    predicted_speedup: float = 1.0
    confidence: float = 0.8
    
    # 分析
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """模拟结果"""
    current_state: CurrentState
    scenarios: List[WhatIfScenario]
    best_scenario: Optional[WhatIfScenario] = None
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_state": {
                "hardware": self.current_state.hardware_name,
                "num_devices": self.current_state.num_devices,
                "parallel_config": f"TP{self.current_state.tp_size}_PP{self.current_state.pp_size}_DP{self.current_state.dp_size}",
                "step_time_ms": self.current_state.step_time_ms,
                "mfu_percent": self.current_state.mfu_percent,
            },
            "scenarios": [
                {
                    "name": s.name,
                    "type": s.scenario_type.value,
                    "description": s.description,
                    "changes": s.changes,
                    "predicted_speedup": s.predicted_speedup,
                    "predicted_mfu": s.predicted_mfu_percent,
                    "confidence": s.confidence,
                    "benefits": s.benefits,
                    "risks": s.risks,
                }
                for s in self.scenarios
            ],
            "best_scenario": self.best_scenario.name if self.best_scenario else None,
            "summary": self.summary,
        }


class WhatIfSimulator:
    """
    What-if 假设分析器
    
    预测不同配置变化对性能的影响
    """
    
    # 硬件升级系数
    HARDWARE_UPGRADE_FACTORS = {
        ("280T", "313T"): 1.12,  # 313/280
        ("280T", "376T"): 1.34,  # 376/280
        ("313T", "376T"): 1.20,  # 376/313
    }
    
    # 精度变化系数（相对于 FP32）
    PRECISION_FACTORS = {
        "fp32": 1.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 4.0,
    }
    
    def __init__(self, current_state: CurrentState):
        """
        初始化模拟器
        
        Args:
            current_state: 当前状态
        """
        self.state = current_state
    
    def simulate_parallel_change(
        self,
        new_tp: int = None,
        new_pp: int = None,
        new_dp: int = None,
    ) -> WhatIfScenario:
        """
        模拟并行度变化
        
        Args:
            new_tp: 新的 TP size
            new_pp: 新的 PP size
            new_dp: 新的 DP size
        """
        old_tp = self.state.tp_size
        old_pp = self.state.pp_size
        old_dp = self.state.dp_size
        
        new_tp = new_tp or old_tp
        new_pp = new_pp or old_pp
        new_dp = new_dp or old_dp
        
        # 验证配置有效性
        total_devices = new_tp * new_pp * new_dp
        if total_devices != self.state.num_devices:
            # 调整 DP 使总数匹配
            new_dp = self.state.num_devices // (new_tp * new_pp)
        
        scenario = WhatIfScenario(
            name=f"并行配置: TP{new_tp}_PP{new_pp}_DP{new_dp}",
            scenario_type=ScenarioType.PARALLEL_CHANGE,
            description=f"从 TP{old_tp}_PP{old_pp}_DP{old_dp} 改为 TP{new_tp}_PP{new_pp}_DP{new_dp}",
            changes={"tp_size": new_tp, "pp_size": new_pp, "dp_size": new_dp},
        )
        
        # 估算性能变化
        speedup = 1.0
        benefits = []
        risks = []
        prerequisites = []
        
        # TP 变化影响
        if new_tp != old_tp:
            if new_tp > old_tp:
                # 增加 TP: 减少单卡计算量，但增加通信
                tp_factor = old_tp / new_tp
                comm_overhead = 1 + 0.1 * (new_tp - old_tp)  # 通信开销增加
                speedup *= tp_factor / comm_overhead
                
                benefits.append(f"单卡显存需求降低 {(1-tp_factor)*100:.0f}%")
                risks.append("TP 通信在关键路径，可能增加延迟")
                
                if new_tp > 8:
                    risks.append("TP > 8 需要跨节点，通信开销显著增加")
            else:
                # 减少 TP
                tp_factor = old_tp / new_tp
                speedup *= 0.95  # 减少通信开销
                benefits.append("减少 TP 通信开销")
                risks.append("单卡显存需求增加")
        
        # PP 变化影响
        if new_pp != old_pp:
            if new_pp > old_pp:
                # 增加 PP
                bubble_ratio = (new_pp - 1) / new_pp
                speedup *= (1 - bubble_ratio * 0.3)  # PP Bubble 影响
                
                benefits.append(f"支持更大模型（每卡只需放 1/{new_pp} 层）")
                risks.append(f"PP Bubble 占比约 {bubble_ratio*30:.0f}%")
                prerequisites.append("需要实现 Pipeline 调度")
            else:
                # 减少 PP
                speedup *= 1.05
                benefits.append("减少 PP Bubble")
                risks.append("单卡显存需求增加")
        
        # DP 变化影响
        if new_dp != old_dp:
            if new_dp > old_dp:
                # 增加 DP: 通信量增加
                comm_factor = 1 + 0.05 * (new_dp - old_dp)
                speedup *= 1 / comm_factor
                
                benefits.append("增加数据并行度，提升吞吐")
                risks.append("梯度同步通信量增加")
            else:
                # 减少 DP
                speedup *= 1.02
                benefits.append("减少 AllReduce 通信")
        
        # 计算预测值
        scenario.predicted_step_time_ms = self.state.step_time_ms / speedup
        scenario.predicted_mfu_percent = self.state.mfu_percent * speedup
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.7 if abs(speedup - 1) > 0.2 else 0.85
        scenario.benefits = benefits
        scenario.risks = risks
        scenario.prerequisites = prerequisites
        
        return scenario
    
    def simulate_hardware_upgrade(
        self,
        new_hardware: str,
    ) -> WhatIfScenario:
        """
        模拟硬件升级
        
        Args:
            new_hardware: 新硬件名称 (如 "376T")
        """
        old_hw = "280T" if "280" in self.state.hardware_name else \
                 "313T" if "313" in self.state.hardware_name else "376T"
        
        scenario = WhatIfScenario(
            name=f"硬件升级: {old_hw} → {new_hardware}",
            scenario_type=ScenarioType.HARDWARE_UPGRADE,
            description=f"从 Atlas A2 {old_hw} 升级到 {new_hardware}",
            changes={"hardware": new_hardware},
        )
        
        # 查找升级系数
        key = (old_hw, new_hardware)
        if key in self.HARDWARE_UPGRADE_FACTORS:
            upgrade_factor = self.HARDWARE_UPGRADE_FACTORS[key]
        elif (new_hardware, old_hw) in self.HARDWARE_UPGRADE_FACTORS:
            upgrade_factor = 1 / self.HARDWARE_UPGRADE_FACTORS[(new_hardware, old_hw)]
        else:
            upgrade_factor = 1.0
        
        # 带宽也会提升
        bandwidth_factors = {
            "280T": 1500,
            "313T": 1800,
            "376T": 2000,
        }
        old_bw = bandwidth_factors.get(old_hw, 1500)
        new_bw = bandwidth_factors.get(new_hardware, old_bw)
        bw_factor = new_bw / old_bw
        
        # 综合考虑算力和带宽
        # 如果是计算受限，主要看算力提升
        # 如果是内存受限，主要看带宽提升
        if self.state.mfu_percent > 40:
            # 可能是计算受限
            speedup = upgrade_factor * 0.8 + bw_factor * 0.2
        else:
            # 可能是内存受限
            speedup = upgrade_factor * 0.5 + bw_factor * 0.5
        
        scenario.predicted_step_time_ms = self.state.step_time_ms / speedup
        scenario.predicted_mfu_percent = self.state.mfu_percent  # MFU 相对于新硬件可能不变
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.9
        
        scenario.benefits = [
            f"算力提升 {(upgrade_factor-1)*100:.0f}%",
            f"带宽提升 {(bw_factor-1)*100:.0f}%",
        ]
        scenario.risks = [
            "需要硬件采购成本",
            "可能需要调整并行配置以充分利用新硬件",
        ]
        scenario.prerequisites = [
            "硬件可用性",
            "驱动和框架兼容性",
        ]
        
        return scenario
    
    def simulate_batch_size_change(
        self,
        new_batch_size: int,
    ) -> WhatIfScenario:
        """
        模拟 Batch Size 变化
        """
        old_bs = self.state.batch_size
        bs_ratio = new_batch_size / old_bs
        
        scenario = WhatIfScenario(
            name=f"Batch Size: {old_bs} → {new_batch_size}",
            scenario_type=ScenarioType.BATCH_SIZE_CHANGE,
            description=f"Batch Size 从 {old_bs} 改为 {new_batch_size}",
            changes={"batch_size": new_batch_size},
        )
        
        # 计算量线性增加，但通信量通常不变（梯度大小固定）
        compute_time_ratio = 1 - self.state.comm_time_ratio
        comm_time_ratio = self.state.comm_time_ratio
        
        # 新的 step 时间
        new_compute_time = compute_time_ratio * self.state.step_time_ms * bs_ratio
        new_comm_time = comm_time_ratio * self.state.step_time_ms
        new_step_time = new_compute_time + new_comm_time
        
        # 吞吐量变化
        old_throughput = old_bs / self.state.step_time_ms
        new_throughput = new_batch_size / new_step_time
        throughput_speedup = new_throughput / old_throughput
        
        # MFU 变化（通常会提升，因为计算密度增加）
        new_mfu = self.state.mfu_percent * (1 + 0.1 * (bs_ratio - 1))
        new_mfu = min(new_mfu, 70)  # 有上限
        
        scenario.predicted_step_time_ms = new_step_time
        scenario.predicted_mfu_percent = new_mfu
        scenario.predicted_speedup = throughput_speedup
        scenario.confidence = 0.8
        
        if new_batch_size > old_bs:
            scenario.benefits = [
                f"吞吐量提升 {(throughput_speedup-1)*100:.0f}%",
                f"MFU 可能提升到 {new_mfu:.0f}%",
                "通信占比相对降低",
            ]
            scenario.risks = [
                "显存需求增加",
                "如果 OOM，需要使用 gradient checkpointing",
                "大 Batch 可能影响收敛",
            ]
            scenario.prerequisites = [
                f"需要约 {bs_ratio:.1f}x 显存",
            ]
        else:
            scenario.benefits = [
                "显存需求降低",
                "更容易适应显存限制",
            ]
            scenario.risks = [
                "吞吐量下降",
                "通信占比相对增加",
            ]
        
        return scenario
    
    def simulate_optimization(
        self,
        optimization: str,
    ) -> WhatIfScenario:
        """
        模拟优化措施
        
        Args:
            optimization: 优化类型
                - "gradient_accumulation": 梯度累积
                - "overlap_optimization": 通信掩盖优化
                - "operator_fusion": 算子融合
                - "mixed_precision": 混合精度
        """
        optimizations = {
            "gradient_accumulation": self._simulate_grad_accum,
            "overlap_optimization": self._simulate_overlap_opt,
            "operator_fusion": self._simulate_op_fusion,
            "mixed_precision": self._simulate_mixed_precision,
        }
        
        if optimization in optimizations:
            return optimizations[optimization]()
        
        return WhatIfScenario(
            name=f"未知优化: {optimization}",
            scenario_type=ScenarioType.OPTIMIZATION,
            description="不支持的优化类型",
            changes={},
        )
    
    def _simulate_grad_accum(self) -> WhatIfScenario:
        """模拟梯度累积"""
        scenario = WhatIfScenario(
            name="梯度累积 (accumulation=4)",
            scenario_type=ScenarioType.OPTIMIZATION,
            description="使用 4 步梯度累积，减少通信频率",
            changes={"gradient_accumulation": 4},
        )
        
        accum_steps = 4
        # 通信频率降低为 1/4
        old_comm_time = self.state.comm_time_ratio * self.state.step_time_ms
        new_comm_time = old_comm_time / accum_steps
        
        # 但每个 micro-step 的计算时间不变
        compute_time = (1 - self.state.comm_time_ratio) * self.state.step_time_ms
        
        # 新的有效 step 时间
        new_step_time = compute_time + new_comm_time
        speedup = self.state.step_time_ms / new_step_time
        
        scenario.predicted_step_time_ms = new_step_time
        scenario.predicted_mfu_percent = self.state.mfu_percent * 1.1
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.85
        
        scenario.benefits = [
            f"通信频率降低 {accum_steps}x",
            "有效 Batch Size 增加",
        ]
        scenario.risks = [
            "需要更多显存存储中间激活值",
            "大 accumulation 可能影响收敛",
        ]
        
        return scenario
    
    def _simulate_overlap_opt(self) -> WhatIfScenario:
        """模拟通信掩盖优化"""
        scenario = WhatIfScenario(
            name="通信掩盖优化",
            scenario_type=ScenarioType.OPTIMIZATION,
            description="优化通信与计算的重叠，目标掩盖率 90%",
            changes={"target_overlap_ratio": 0.9},
        )
        
        old_overlap = self.state.overlap_ratio
        new_overlap = 0.9
        
        # 计算时间节省
        comm_time = self.state.comm_time_ratio * self.state.step_time_ms
        old_exposed_comm = comm_time * (1 - old_overlap)
        new_exposed_comm = comm_time * (1 - new_overlap)
        
        time_saved = old_exposed_comm - new_exposed_comm
        new_step_time = self.state.step_time_ms - time_saved
        speedup = self.state.step_time_ms / new_step_time
        
        scenario.predicted_step_time_ms = new_step_time
        scenario.predicted_mfu_percent = self.state.mfu_percent * speedup
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.75
        
        scenario.benefits = [
            f"暴露通信时间减少 {(old_exposed_comm - new_exposed_comm)/1000:.1f}ms",
            f"Step 时间减少 {time_saved/1000:.1f}ms",
        ]
        scenario.risks = [
            "可能需要修改训练代码",
            "某些框架不支持完全异步通信",
        ]
        scenario.prerequisites = [
            "使用支持异步通信的框架",
            "合理的计算/通信调度",
        ]
        
        return scenario
    
    def _simulate_op_fusion(self) -> WhatIfScenario:
        """模拟算子融合"""
        scenario = WhatIfScenario(
            name="算子融合优化",
            scenario_type=ScenarioType.OPTIMIZATION,
            description="融合相邻小算子，减少 Kernel Launch 和内存访问",
            changes={"operator_fusion": True},
        )
        
        # 假设融合能带来 5-15% 的提升
        speedup = 1.10
        
        scenario.predicted_step_time_ms = self.state.step_time_ms / speedup
        scenario.predicted_mfu_percent = self.state.mfu_percent * 1.08
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.7
        
        scenario.benefits = [
            "减少 Kernel Launch 开销",
            "减少中间 Tensor 的内存写回",
            "提高 L2 Cache 利用率",
        ]
        scenario.risks = [
            "需要编译器/框架支持",
            "可能增加编译时间",
        ]
        scenario.prerequisites = [
            "使用支持算子融合的框架（如 TorchInductor, MindSpeed）",
        ]
        
        return scenario
    
    def _simulate_mixed_precision(self) -> WhatIfScenario:
        """模拟混合精度"""
        current_precision = self.state.precision
        
        if current_precision == "fp32":
            new_precision = "bf16"
            speedup = 1.8  # FP32 -> BF16 约 1.8x
        else:
            # 已经是混合精度
            new_precision = current_precision
            speedup = 1.0
        
        scenario = WhatIfScenario(
            name=f"混合精度: {current_precision} → {new_precision}",
            scenario_type=ScenarioType.PRECISION_CHANGE,
            description=f"从 {current_precision} 切换到 {new_precision} 混合精度训练",
            changes={"precision": new_precision},
        )
        
        scenario.predicted_step_time_ms = self.state.step_time_ms / speedup
        scenario.predicted_mfu_percent = self.state.mfu_percent * speedup * 0.9
        scenario.predicted_speedup = speedup
        scenario.confidence = 0.9 if speedup > 1 else 1.0
        
        if speedup > 1:
            scenario.benefits = [
                f"计算速度提升约 {(speedup-1)*100:.0f}%",
                "显存占用降低约 50%",
            ]
            scenario.risks = [
                "可能需要调整学习率和 loss scale",
                "某些模型对精度敏感",
            ]
            scenario.prerequisites = [
                "确认模型支持混合精度训练",
                "配置合适的 loss scaling",
            ]
        else:
            scenario.benefits = ["已使用混合精度，无需变更"]
        
        return scenario
    
    def run_all_scenarios(self) -> SimulationResult:
        """
        运行所有预定义场景
        
        Returns:
            SimulationResult: 完整的模拟结果
        """
        scenarios = []
        
        # 并行度变化场景
        parallel_configs = [
            (2, 1, None),   # 增加 TP
            (4, 1, None),   # 更大 TP
            (1, 2, None),   # 增加 PP
            (2, 2, None),   # TP+PP
        ]
        for tp, pp, dp in parallel_configs:
            scenario = self.simulate_parallel_change(tp, pp, dp)
            scenarios.append(scenario)
        
        # 硬件升级场景
        for hw in ["313T", "376T"]:
            if hw not in self.state.hardware_name:
                scenario = self.simulate_hardware_upgrade(hw)
                scenarios.append(scenario)
        
        # Batch Size 变化
        for bs in [self.state.batch_size * 2, self.state.batch_size // 2]:
            if bs > 0:
                scenario = self.simulate_batch_size_change(bs)
                scenarios.append(scenario)
        
        # 优化措施
        for opt in ["gradient_accumulation", "overlap_optimization", "operator_fusion"]:
            scenario = self.simulate_optimization(opt)
            scenarios.append(scenario)
        
        # 混合精度
        if self.state.precision == "fp32":
            scenario = self.simulate_optimization("mixed_precision")
            scenarios.append(scenario)
        
        # 找出最佳场景
        valid_scenarios = [s for s in scenarios if s.predicted_speedup > 1]
        best_scenario = max(valid_scenarios, key=lambda s: s.predicted_speedup * s.confidence) if valid_scenarios else None
        
        # 生成总结
        if best_scenario:
            summary = f"推荐优化方案: {best_scenario.name}，预计加速 {best_scenario.predicted_speedup:.1f}x"
        else:
            summary = "当前配置已接近最优，无明显优化空间"
        
        return SimulationResult(
            current_state=self.state,
            scenarios=sorted(scenarios, key=lambda s: -s.predicted_speedup),
            best_scenario=best_scenario,
            summary=summary,
        )
