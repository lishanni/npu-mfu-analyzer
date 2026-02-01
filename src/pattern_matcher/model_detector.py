"""
模型结构检测器

从 Profiling 数据推断模型架构参数（层数、hidden size、attention heads 等）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """模型架构类型"""
    TRANSFORMER = "transformer"
    GPT = "gpt"
    BERT = "bert"
    LLAMA = "llama"
    MIXTRAL = "mixtral"  # MoE
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: ModelArchitecture = ModelArchitecture.UNKNOWN
    
    # 核心参数
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None  # FFN hidden size
    vocab_size: Optional[int] = None
    max_seq_length: Optional[int] = None
    
    # 高级参数
    num_key_value_heads: Optional[int] = None  # GQA/MQA
    num_experts: Optional[int] = None  # MoE
    experts_per_token: Optional[int] = None  # MoE routing
    
    # 检测置信度和证据
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        parts = [f"Architecture: {self.architecture.value}"]
        if self.num_layers:
            parts.append(f"Layers: {self.num_layers}")
        if self.hidden_size:
            parts.append(f"Hidden: {self.hidden_size}")
        if self.num_attention_heads:
            parts.append(f"Heads: {self.num_attention_heads}")
        if self.intermediate_size:
            parts.append(f"FFN: {self.intermediate_size}")
        return ", ".join(parts)


class ModelDetector:
    """
    模型结构检测器
    
    通过分析算子名称、张量形状、计算模式等，推断模型架构和参数。
    """
    
    # 常见模型架构特征
    ARCHITECTURE_SIGNATURES = {
        ModelArchitecture.LLAMA: {
            "operator_patterns": [r".*llama.*", r".*rotary.*"],
            "features": ["rotary_embedding", "rms_norm"],
        },
        ModelArchitecture.GPT: {
            "operator_patterns": [r".*gpt.*", r".*causal.*"],
            "features": ["layer_norm", "causal_attention"],
        },
        ModelArchitecture.BERT: {
            "operator_patterns": [r".*bert.*"],
            "features": ["layer_norm", "bidirectional_attention"],
        },
        ModelArchitecture.MIXTRAL: {
            "operator_patterns": [r".*mixtral.*", r".*moe.*", r".*expert.*"],
            "features": ["moe_router", "sparse_expert"],
        },
    }
    
    def __init__(self):
        pass
    
    def detect(
        self,
        operator_names: List[str],
        operator_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> ModelConfig:
        """
        检测模型配置
        
        Args:
            operator_names: 算子名称列表
            operator_shapes: 算子张量形状字典 {op_name: [shape]}
            
        Returns:
            ModelConfig
        """
        if operator_shapes is None:
            operator_shapes = {}
        
        evidence = []
        
        # 1. 检测架构类型
        architecture = self._detect_architecture(operator_names, evidence)
        
        # 2. 从算子名称推断参数
        num_layers = self._detect_num_layers(operator_names, evidence)
        hidden_size = self._detect_hidden_size(operator_names, operator_shapes, evidence)
        num_heads = self._detect_num_heads(operator_names, operator_shapes, evidence)
        intermediate_size = self._detect_intermediate_size(operator_names, operator_shapes, evidence)
        
        # 3. 检测 MoE 参数
        num_experts = self._detect_num_experts(operator_names, evidence)
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(evidence)
        
        return ModelConfig(
            architecture=architecture,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            confidence=confidence,
            evidence=evidence,
        )
    
    def _detect_architecture(
        self,
        operator_names: List[str],
        evidence: List[str]
    ) -> ModelArchitecture:
        """检测模型架构类型"""
        scores = {arch: 0 for arch in ModelArchitecture}
        
        for arch, signature in self.ARCHITECTURE_SIGNATURES.items():
            for pattern in signature["operator_patterns"]:
                for op_name in operator_names:
                    if re.search(pattern, op_name, re.IGNORECASE):
                        scores[arch] += 1
                        evidence.append(f"Architecture signature matched: {arch.value}")
                        break
        
        best_arch = max(scores, key=scores.get)
        if scores[best_arch] == 0:
            return ModelArchitecture.TRANSFORMER  # 默认为 Transformer
        
        return best_arch
    
    def _detect_num_layers(
        self,
        operator_names: List[str],
        evidence: List[str]
    ) -> Optional[int]:
        """
        检测层数
        
        通过算子名称中的层索引推断，如：
        - "layer.0.attention"
        - "transformer.h.23.mlp"
        """
        layer_indices = set()
        
        patterns = [
            r"layer[._](\d+)",
            r"layers[._](\d+)",
            r"h[._](\d+)",
            r"block[._](\d+)",
        ]
        
        for op_name in operator_names:
            for pattern in patterns:
                match = re.search(pattern, op_name, re.IGNORECASE)
                if match:
                    layer_idx = int(match.group(1))
                    layer_indices.add(layer_idx)
        
        if layer_indices:
            # 层数 = 最大索引 + 1（假设从 0 开始）
            num_layers = max(layer_indices) + 1
            evidence.append(f"Detected {num_layers} layers from operator names")
            return num_layers
        
        return None
    
    def _detect_hidden_size(
        self,
        operator_names: List[str],
        operator_shapes: Dict[str, List[int]],
        evidence: List[str]
    ) -> Optional[int]:
        """
        检测 hidden size
        
        通过 MatMul/Linear 算子的权重形状推断
        """
        # 查找 attention 相关的 MatMul
        for op_name, shapes in operator_shapes.items():
            if "attention" in op_name.lower() or "qkv" in op_name.lower():
                if shapes and len(shapes) >= 2:
                    # 假设形状为 [hidden_size, ...]
                    hidden_size = shapes[0]
                    if 512 <= hidden_size <= 16384:  # 合理范围
                        evidence.append(f"Detected hidden_size={hidden_size} from {op_name}")
                        return hidden_size
        
        # 从算子名称中提取（如 "hidden_4096"）
        for op_name in operator_names:
            match = re.search(r"hidden[._]?(\d+)", op_name, re.IGNORECASE)
            if match:
                hidden_size = int(match.group(1))
                evidence.append(f"Detected hidden_size={hidden_size} from operator name")
                return hidden_size
        
        return None
    
    def _detect_num_heads(
        self,
        operator_names: List[str],
        operator_shapes: Dict[str, List[int]],
        evidence: List[str]
    ) -> Optional[int]:
        """检测注意力头数"""
        # 从算子名称中提取
        for op_name in operator_names:
            match = re.search(r"heads?[._]?(\d+)", op_name, re.IGNORECASE)
            if match:
                num_heads = int(match.group(1))
                evidence.append(f"Detected {num_heads} attention heads from operator name")
                return num_heads
        
        # 从 attention 算子形状推断
        for op_name, shapes in operator_shapes.items():
            if "attention" in op_name.lower() and shapes:
                # 典型形状: [batch, num_heads, seq_len, head_dim]
                if len(shapes) >= 4:
                    num_heads = shapes[1]
                    if 4 <= num_heads <= 128:  # 合理范围
                        evidence.append(f"Detected {num_heads} heads from shape in {op_name}")
                        return num_heads
        
        return None
    
    def _detect_intermediate_size(
        self,
        operator_names: List[str],
        operator_shapes: Dict[str, List[int]],
        evidence: List[str]
    ) -> Optional[int]:
        """检测 FFN 中间层大小"""
        # 查找 FFN/MLP 相关的算子
        for op_name, shapes in operator_shapes.items():
            if ("mlp" in op_name.lower() or "ffn" in op_name.lower() or "fc" in op_name.lower()):
                if shapes and len(shapes) >= 2:
                    # 通常是 hidden_size -> intermediate_size 的投影
                    intermediate_size = max(shapes)
                    if 1024 <= intermediate_size <= 65536:  # 合理范围
                        evidence.append(f"Detected intermediate_size={intermediate_size} from {op_name}")
                        return intermediate_size
        
        return None
    
    def _detect_num_experts(
        self,
        operator_names: List[str],
        evidence: List[str]
    ) -> Optional[int]:
        """检测 MoE 专家数量"""
        expert_indices = set()
        
        patterns = [
            r"expert[._](\d+)",
            r"experts[._](\d+)",
        ]
        
        for op_name in operator_names:
            for pattern in patterns:
                match = re.search(pattern, op_name, re.IGNORECASE)
                if match:
                    expert_idx = int(match.group(1))
                    expert_indices.add(expert_idx)
        
        if expert_indices:
            num_experts = max(expert_indices) + 1
            evidence.append(f"Detected {num_experts} experts (MoE) from operator names")
            return num_experts
        
        return None
    
    def _calculate_confidence(self, evidence: List[str]) -> float:
        """计算检测置信度"""
        if not evidence:
            return 0.0
        
        # 基于证据数量
        confidence = min(len(evidence) / 4.0, 1.0)  # 4 条证据 = 100%
        return round(confidence, 2)


def detect_model_config_from_loader(loader) -> ModelConfig:
    """
    从 ProfilingLoader 检测模型配置
    
    Args:
        loader: ProfilingLoader 实例
        
    Returns:
        ModelConfig
    """
    detector = ModelDetector()
    
    # 收集算子名称
    operator_names = []
    operator_shapes = {}
    
    try:
        # 从 DB 获取算子信息
        # 需要根据实际的 ProfilingLoader 接口实现
        pass
    except Exception as e:
        logger.debug(f"Failed to collect operator info: {e}")
    
    return detector.detect(operator_names, operator_shapes)
