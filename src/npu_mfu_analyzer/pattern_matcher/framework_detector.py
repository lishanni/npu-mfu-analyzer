"""
框架检测器

自动识别训练框架类型（Megatron-LM、DeepSpeed、HuggingFace Accelerate、PyTorch DDP 等）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class FrameworkType(Enum):
    """支持的框架类型"""
    MEGATRON = "megatron"
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    FSDP2 = "fsdp2"
    ACCELERATE = "accelerate"
    PYTORCH_DDP = "pytorch_ddp"
    MINDSPORE = "mindspore"
    MINDSPEED = "mindspeed"
    TITAN = "titan"
    UNKNOWN = "unknown"


@dataclass
class FrameworkSignature:
    """框架特征签名"""
    name: FrameworkType
    # 算子名称模式
    operator_patterns: List[str]
    # 通信组命名模式
    comm_group_patterns: List[str]
    # 特征性的算子组合
    op_sequence_patterns: List[List[str]]
    # 环境变量或配置特征
    env_patterns: List[str]
    # 置信度权重
    confidence_weight: Dict[str, float]


@dataclass
class DetectionResult:
    """检测结果"""
    framework: FrameworkType
    confidence: float
    evidence: List[str]
    version: Optional[str] = None
    config_hints: Dict[str, Any] = None


class FrameworkDetector:
    """
    框架检测器
    
    通过分析 Profiling 数据中的算子名称、通信组名称、算子序列等特征，
    自动识别使用的训练框架。
    """
    
    # 预定义框架签名库
    SIGNATURES = [
        FrameworkSignature(
            name=FrameworkType.MEGATRON,
            operator_patterns=[
                r".*megatron.*",
                r".*parallel_embedding.*",
                r".*column_parallel.*",
                r".*row_parallel.*",
            ],
            comm_group_patterns=[
                r".*tensor_model_parallel.*",
                r".*pipeline_model_parallel.*",
                r".*data_parallel.*",
            ],
            op_sequence_patterns=[
                ["column_parallel_linear", "allreduce", "row_parallel_linear"],
            ],
            env_patterns=[
                "MEGATRON_",
                "NPROC_PER_NODE",
            ],
            confidence_weight={
                "operator": 0.4,
                "comm_group": 0.3,
                "sequence": 0.2,
                "env": 0.1,
            }
        ),
        FrameworkSignature(
            name=FrameworkType.DEEPSPEED,
            operator_patterns=[
                r".*deepspeed.*",
                r".*ds_.*",
                r".*zero.*optimizer.*",
            ],
            comm_group_patterns=[
                r".*deepspeed.*",
                r".*zero_stage.*",
            ],
            op_sequence_patterns=[
                ["reducescatter", "allgather"],  # ZeRO Stage 2/3
            ],
            env_patterns=[
                "DEEPSPEED_",
                "DS_",
            ],
            confidence_weight={
                "operator": 0.35,
                "comm_group": 0.35,
                "sequence": 0.2,
                "env": 0.1,
            }
        ),
        FrameworkSignature(
            name=FrameworkType.FSDP,
            operator_patterns=[
                r".*fsdp.*",
                r".*fully_sharded.*",
            ],
            comm_group_patterns=[
                r".*fsdp.*",
            ],
            op_sequence_patterns=[
                ["allgather", "compute", "reducescatter"],  # FSDP 典型模式
            ],
            env_patterns=[
                "FSDP_",
            ],
            confidence_weight={
                "operator": 0.4,
                "comm_group": 0.3,
                "sequence": 0.2,
                "env": 0.1,
            }
        ),
        FrameworkSignature(
            name=FrameworkType.PYTORCH_DDP,
            operator_patterns=[
                r".*distributed.*data.*parallel.*",
                r".*ddp.*",
            ],
            comm_group_patterns=[
                r".*ddp.*",
                r"^default$",  # DDP 通常使用默认通信组
            ],
            op_sequence_patterns=[
                ["backward", "allreduce"],  # DDP 梯度同步
            ],
            env_patterns=[
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
            ],
            confidence_weight={
                "operator": 0.3,
                "comm_group": 0.2,
                "sequence": 0.3,
                "env": 0.2,
            }
        ),
        FrameworkSignature(
            name=FrameworkType.MINDSPEED,
            operator_patterns=[
                r".*mindspeed.*",
                r".*ms_.*",
            ],
            comm_group_patterns=[
                r".*mindspeed.*",
            ],
            op_sequence_patterns=[],
            env_patterns=[
                "MINDSPEED_",
            ],
            confidence_weight={
                "operator": 0.5,
                "comm_group": 0.3,
                "sequence": 0.1,
                "env": 0.1,
            }
        ),
    ]
    
    def __init__(self):
        self.signatures = self.SIGNATURES
    
    def detect(
        self,
        operator_names: List[str],
        comm_groups: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> DetectionResult:
        """
        检测框架类型
        
        Args:
            operator_names: 算子名称列表
            comm_groups: 通信组名称列表
            env_vars: 环境变量字典
            
        Returns:
            DetectionResult
        """
        if comm_groups is None:
            comm_groups = []
        if env_vars is None:
            env_vars = {}
        
        scores = {}
        evidence_map = {}
        
        for signature in self.signatures:
            score, evidence = self._calculate_signature_score(
                signature, operator_names, comm_groups, env_vars
            )
            scores[signature.name] = score
            evidence_map[signature.name] = evidence
        
        # 找到得分最高的框架
        best_framework = max(scores, key=scores.get)
        best_score = scores[best_framework]
        
        # 如果最高分太低，标记为 UNKNOWN
        if best_score < 0.3:
            best_framework = FrameworkType.UNKNOWN
            best_score = 0.0
            evidence = ["No clear framework signature detected"]
        else:
            evidence = evidence_map[best_framework]
        
        return DetectionResult(
            framework=best_framework,
            confidence=best_score,
            evidence=evidence,
        )
    
    def _calculate_signature_score(
        self,
        signature: FrameworkSignature,
        operator_names: List[str],
        comm_groups: List[str],
        env_vars: Dict[str, str],
    ) -> tuple[float, List[str]]:
        """计算签名匹配得分"""
        evidence = []
        component_scores = {}
        
        # 1. 算子名称匹配
        op_matches = 0
        for pattern in signature.operator_patterns:
            for op_name in operator_names:
                if re.search(pattern, op_name, re.IGNORECASE):
                    op_matches += 1
                    evidence.append(f"Operator pattern matched: {pattern}")
                    break
        
        if signature.operator_patterns:
            component_scores["operator"] = op_matches / len(signature.operator_patterns)
        else:
            component_scores["operator"] = 0.0
        
        # 2. 通信组匹配
        comm_matches = 0
        for pattern in signature.comm_group_patterns:
            for group in comm_groups:
                if re.search(pattern, group, re.IGNORECASE):
                    comm_matches += 1
                    evidence.append(f"Comm group pattern matched: {pattern}")
                    break
        
        if signature.comm_group_patterns:
            component_scores["comm_group"] = comm_matches / len(signature.comm_group_patterns)
        else:
            component_scores["comm_group"] = 0.0
        
        # 3. 算子序列匹配（简化实现，实际需要更复杂的序列匹配）
        seq_matches = 0
        for seq_pattern in signature.op_sequence_patterns:
            # 简化：检查序列中的关键词是否在算子列表中
            if all(any(keyword in op.lower() for op in operator_names) for keyword in seq_pattern):
                seq_matches += 1
                evidence.append(f"Operator sequence matched: {' -> '.join(seq_pattern)}")
        
        if signature.op_sequence_patterns:
            component_scores["sequence"] = seq_matches / len(signature.op_sequence_patterns)
        else:
            component_scores["sequence"] = 0.0
        
        # 4. 环境变量匹配
        env_matches = 0
        for env_pattern in signature.env_patterns:
            for env_key in env_vars:
                if env_pattern in env_key:
                    env_matches += 1
                    evidence.append(f"Environment variable matched: {env_pattern}")
                    break
        
        if signature.env_patterns:
            component_scores["env"] = env_matches / len(signature.env_patterns)
        else:
            component_scores["env"] = 0.0
        
        # 加权总分
        total_score = sum(
            component_scores.get(key, 0) * weight
            for key, weight in signature.confidence_weight.items()
        )
        
        return total_score, evidence
    
    def detect_from_profiling_data(self, profiling_data: Dict[str, Any]) -> DetectionResult:
        """
        从 Profiling 数据检测框架
        
        Args:
            profiling_data: Profiling 数据字典，可能包含：
                - operators: 算子列表
                - comm_events: 通信事件列表
                - environment: 环境变量
                
        Returns:
            DetectionResult
        """
        operator_names = []
        if "operators" in profiling_data:
            operator_names = [op.get("name", "") for op in profiling_data["operators"]]
        
        comm_groups = []
        if "comm_events" in profiling_data:
            for event in profiling_data["comm_events"]:
                group_name = event.get("args", {}).get("group_name") or event.get("args", {}).get("groupName")
                if group_name and group_name not in comm_groups:
                    comm_groups.append(str(group_name))
        
        env_vars = profiling_data.get("environment", {})
        
        return self.detect(operator_names, comm_groups, env_vars)


def detect_framework_from_loader(loader) -> DetectionResult:
    """
    从 ProfilingLoader 检测框架
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        DetectionResult
    """
    detector = FrameworkDetector()
    
    # 收集算子名称
    operator_names = []
    try:
        # 从 DB 或其他来源获取算子名称
        # 这里需要根据 ProfilingLoader 的实际接口实现
        pass
    except Exception as e:
        logger.debug(f"Failed to collect operator names: {e}")
    
    # 收集通信组
    comm_groups = []
    try:
        comm_data = loader.get_communication_data()
        if not comm_data.empty and "groupName" in comm_data.columns:
            comm_groups = comm_data["groupName"].unique().tolist()
    except Exception as e:
        logger.debug(f"Failed to collect comm groups: {e}")
    
    return detector.detect(operator_names, comm_groups)
