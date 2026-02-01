"""
Universal Pattern Matcher

跨框架模式识别的统一入口，整合框架检测、并行配置检测和模型结构检测
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging

from src.pattern_matcher.framework_detector import (
    FrameworkDetector,
    DetectionResult as FrameworkDetectionResult,
)
from src.pattern_matcher.parallel_detector import (
    ParallelDetector,
    ParallelConfig,
)
from src.pattern_matcher.model_detector import (
    ModelDetector,
    ModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class UniversalPattern:
    """通用模式检测结果"""
    framework: FrameworkDetectionResult
    parallel_config: ParallelConfig
    model_config: ModelConfig
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "framework": {
                "type": self.framework.framework.value,
                "confidence": self.framework.confidence,
                "version": self.framework.version,
                "evidence": self.framework.evidence,
            },
            "parallel": {
                "world_size": self.parallel_config.world_size,
                "tp_size": self.parallel_config.tensor_parallel_size,
                "pp_size": self.parallel_config.pipeline_parallel_size,
                "dp_size": self.parallel_config.data_parallel_size,
                "cp_size": self.parallel_config.context_parallel_size,
                "ep_size": self.parallel_config.expert_parallel_size,
                "use_zero": self.parallel_config.use_zero,
                "zero_stage": self.parallel_config.zero_stage,
                "use_fsdp": self.parallel_config.use_fsdp,
                "confidence": self.parallel_config.confidence,
                "evidence": self.parallel_config.evidence,
            },
            "model": {
                "architecture": self.model_config.architecture.value,
                "num_layers": self.model_config.num_layers,
                "hidden_size": self.model_config.hidden_size,
                "num_attention_heads": self.model_config.num_attention_heads,
                "intermediate_size": self.model_config.intermediate_size,
                "num_experts": self.model_config.num_experts,
                "confidence": self.model_config.confidence,
                "evidence": self.model_config.evidence,
            },
        }
    
    def to_prompt_text(self) -> str:
        """转换为 LLM Prompt 格式"""
        lines = [
            "## 训练配置识别结果",
            "",
            "### 框架信息",
            f"- **框架类型**: {self.framework.framework.value}",
            f"- **置信度**: {self.framework.confidence:.0%}",
        ]
        
        if self.framework.version:
            lines.append(f"- **版本**: {self.framework.version}")
        
        if self.framework.evidence:
            lines.append("- **检测依据**:")
            for evidence in self.framework.evidence[:3]:  # 只显示前 3 条
                lines.append(f"  - {evidence}")
        
        lines.extend([
            "",
            "### 并行策略",
            f"- **总进程数**: {self.parallel_config.world_size}",
        ])
        
        if self.parallel_config.tensor_parallel_size > 1:
            lines.append(f"- **Tensor Parallel (TP)**: {self.parallel_config.tensor_parallel_size}")
        
        if self.parallel_config.pipeline_parallel_size > 1:
            lines.append(f"- **Pipeline Parallel (PP)**: {self.parallel_config.pipeline_parallel_size}")
        
        if self.parallel_config.data_parallel_size > 1:
            lines.append(f"- **Data Parallel (DP)**: {self.parallel_config.data_parallel_size}")
        
        if self.parallel_config.context_parallel_size > 1:
            lines.append(f"- **Context Parallel (CP)**: {self.parallel_config.context_parallel_size}")
        
        if self.parallel_config.expert_parallel_size > 1:
            lines.append(f"- **Expert Parallel (EP)**: {self.parallel_config.expert_parallel_size}")
        
        if self.parallel_config.use_zero:
            stage_info = f" Stage {self.parallel_config.zero_stage}" if self.parallel_config.zero_stage else ""
            lines.append(f"- **ZeRO 优化**: 启用{stage_info}")
        
        if self.parallel_config.use_fsdp:
            lines.append("- **FSDP**: 启用")
        
        lines.append(f"- **置信度**: {self.parallel_config.confidence:.0%}")
        
        lines.extend([
            "",
            "### 模型架构",
            f"- **架构类型**: {self.model_config.architecture.value}",
        ])
        
        if self.model_config.num_layers:
            lines.append(f"- **层数**: {self.model_config.num_layers}")
        
        if self.model_config.hidden_size:
            lines.append(f"- **Hidden Size**: {self.model_config.hidden_size}")
        
        if self.model_config.num_attention_heads:
            lines.append(f"- **Attention Heads**: {self.model_config.num_attention_heads}")
        
        if self.model_config.intermediate_size:
            lines.append(f"- **FFN Intermediate Size**: {self.model_config.intermediate_size}")
        
        if self.model_config.num_experts:
            lines.append(f"- **MoE Experts**: {self.model_config.num_experts}")
        
        lines.append(f"- **置信度**: {self.model_config.confidence:.0%}")
        
        return "\n".join(lines)


class UniversalPatternMatcher:
    """
    通用模式匹配器
    
    整合框架检测、并行配置检测和模型结构检测，
    提供统一的接口进行跨框架模式识别。
    """
    
    def __init__(self):
        self.framework_detector = FrameworkDetector()
        self.model_detector = ModelDetector()
    
    def detect(
        self,
        profiling_loader,
        operator_names: Optional[list] = None,
        comm_events: Optional[list] = None,
    ) -> UniversalPattern:
        """
        执行全面的模式检测
        
        Args:
            profiling_loader: ProfilingLoader 实例
            operator_names: 算子名称列表（可选，从 loader 自动提取）
            comm_events: 通信事件列表（可选，从 loader 自动提取）
            
        Returns:
            UniversalPattern: 检测结果
        """
        # 1. 收集数据
        if operator_names is None:
            operator_names = self._extract_operator_names(profiling_loader)
        
        if comm_events is None:
            comm_events = self._extract_comm_events(profiling_loader)
        
        comm_groups = self._extract_comm_groups(comm_events)
        
        # 2. 框架检测
        logger.info("Detecting training framework...")
        framework_result = self.framework_detector.detect(
            operator_names=operator_names,
            comm_groups=comm_groups,
        )
        logger.info(
            f"Framework detected: {framework_result.framework.value} "
            f"(confidence: {framework_result.confidence:.0%})"
        )
        
        # 3. 并行配置检测
        logger.info("Detecting parallel configuration...")
        info = profiling_loader.detect()
        parallel_detector = ParallelDetector(world_size=info.rank_count)
        parallel_config = parallel_detector.detect(
            comm_events=comm_events,
            operator_names=operator_names,
        )
        logger.info(
            f"Parallel config detected: TP={parallel_config.tensor_parallel_size}, "
            f"PP={parallel_config.pipeline_parallel_size}, "
            f"DP={parallel_config.data_parallel_size}"
        )
        
        # 4. 模型结构检测
        logger.info("Detecting model architecture...")
        model_config = self.model_detector.detect(
            operator_names=operator_names,
        )
        logger.info(
            f"Model detected: {model_config.architecture.value} "
            f"(layers={model_config.num_layers}, hidden={model_config.hidden_size})"
        )
        
        return UniversalPattern(
            framework=framework_result,
            parallel_config=parallel_config,
            model_config=model_config,
        )
    
    def _extract_operator_names(self, loader) -> list:
        """从 ProfilingLoader 提取算子名称"""
        operator_names = []
        
        try:
            # 尝试从数据摘要获取
            summary = loader.get_timeline_summary()
            if "raw_data" in summary:
                for record in summary["raw_data"]:
                    if "name" in record:
                        operator_names.append(record["name"])
        except Exception as e:
            logger.debug(f"Failed to extract operator names: {e}")
        
        return operator_names
    
    def _extract_comm_events(self, loader) -> list:
        """从 ProfilingLoader 提取通信事件"""
        comm_events = []
        
        try:
            overlap_events = loader.get_overlap_events()
            comm_events = overlap_events.get("hccl", [])
        except Exception as e:
            logger.debug(f"Failed to extract comm events: {e}")
        
        return comm_events
    
    def _extract_comm_groups(self, comm_events: list) -> list:
        """从通信事件提取通信组名称"""
        comm_groups = []
        
        for event in comm_events:
            args = event.get("args", {})
            group_name = args.get("group_name") or args.get("groupName")
            if group_name and group_name not in comm_groups:
                comm_groups.append(str(group_name))
        
        return comm_groups


def detect_pattern_from_loader(loader) -> UniversalPattern:
    """
    从 ProfilingLoader 检测所有模式
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        UniversalPattern
    """
    matcher = UniversalPatternMatcher()
    return matcher.detect(loader)
