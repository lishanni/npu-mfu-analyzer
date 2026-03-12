"""
堆栈数据结构定义

用于 Host-Device 关联分析的数据类型。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class StackFrame:
    """单个堆栈帧"""
    function: str        # 函数名，如 "CompiledFunctionBackward"
    filename: str        # 文件路径
    lineno: int          # 行号

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function,
            "filename": self.filename,
            "lineno": self.lineno,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StackFrame":
        return cls(
            function=data.get("function", ""),
            filename=data.get("filename", ""),
            lineno=data.get("lineno", 0),
        )

    def __str__(self) -> str:
        return f"{self.function} ({self.filename}:{self.lineno})"


@dataclass
class HostStack:
    """Host 侧完整堆栈"""
    python_stack: List[StackFrame] = field(default_factory=list)   # Python 堆栈
    cpp_stack: List[StackFrame] = field(default_factory=list)      # C++ 堆栈（可选）

    # 识别特征
    is_torch_compile: bool = False           # 是否来自 torch.compile
    is_eager: bool = False                   # 是否来自 eager 模式
    is_fusion_op: bool = False               # 是否来自融合算子
    fusion_op_name: Optional[str] = None     # 融合算子名称
    is_mindspeed: bool = False               # 是否来自 mindspeed
    is_torch_ascend: bool = False            # 是否来自 torch-ascend/CANN
    is_distributed: bool = False             # 是否来自分布式通信
    is_optimizer: bool = False               # 是否来自优化器

    # 原始堆栈字符串（用于调试）
    raw_python_stack: Optional[str] = None
    raw_cpp_stack: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_stack": [f.to_dict() for f in self.python_stack],
            "cpp_stack": [f.to_dict() for f in self.cpp_stack],
            "is_torch_compile": self.is_torch_compile,
            "is_eager": self.is_eager,
            "is_fusion_op": self.is_fusion_op,
            "fusion_op_name": self.fusion_op_name,
            "is_mindspeed": self.is_mindspeed,
            "is_torch_ascend": self.is_torch_ascend,
            "is_distributed": self.is_distributed,
            "is_optimizer": self.is_optimizer,
        }

    def get_source_type(self) -> str:
        """
        获取算子来源类型

        Returns:
            "torch_compile" / "eager" / "fusion_op" / "mindspeed" / "torch_ascend" / "distributed" / "optimizer" / "unknown"
        """
        if self.is_fusion_op:
            return "fusion_op"
        if self.is_torch_compile:
            return "torch_compile"
        if self.is_mindspeed:
            return "mindspeed"
        if self.is_torch_ascend:
            return "torch_ascend"
        if self.is_distributed:
            return "distributed"
        if self.is_optimizer:
            return "optimizer"
        if self.is_eager:
            return "eager"
        return "unknown"

    def get_top_frames(self, n: int = 5) -> List[str]:
        """获取前 N 帧的函数名"""
        return [f.function for f in self.python_stack[:n]]

    def __str__(self) -> str:
        lines = []
        if self.python_stack:
            lines.append("Python Stack:")
            for i, frame in enumerate(self.python_stack[:10]):
                lines.append(f"  {i}: {frame}")
        if self.cpp_stack:
            lines.append("C++ Stack:")
            for i, frame in enumerate(self.cpp_stack[:5]):
                lines.append(f"  {i}: {frame}")
        return "\n".join(lines)


@dataclass
class OperatorWithStack:
    """带堆栈信息的算子"""
    name: str
    dur: float          # 耗时 (us)
    cat: str            # category
    ts: float           # 开始时间戳
    pid: Any            # process id
    tid: Any            # thread id

    # 关联的 Host 堆栈
    host_stack: Optional[HostStack] = None

    # 关联信息（用于 Host-Device 关联）
    connection_id: Optional[int] = None

    # 原始事件数据（可选）
    raw_event: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dur": self.dur,
            "cat": self.cat,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
            "host_stack": self.host_stack.to_dict() if self.host_stack else None,
            "connection_id": self.connection_id,
            "source_type": self.host_stack.get_source_type() if self.host_stack else "unknown",
        }

    def get_source_type(self) -> str:
        """获取算子来源类型"""
        if self.host_stack:
            return self.host_stack.get_source_type()
        return "unknown"


@dataclass
class HostDeviceChain:
    """
    Host 到 Device 的完整调用链

    基于 msprof 的 connection_id 机制建立关联
    """
    # Host 侧 - Torch 算子
    torch_op_name: str = ""
    torch_op_ts: float = 0.0          # Torch 算子开始时间
    torch_op_dur: float = 0.0         # Torch 算子耗时 (us)

    # CANN 层 - ACL API
    acl_api_name: str = ""
    acl_api_ts: float = 0.0           # ACL API 开始时间
    acl_api_dur: float = 0.0          # ACL API 耗时 (us)

    # Device 侧 - NPU 算子
    device_op_name: str = ""
    device_op_ts: float = 0.0         # Device 算子开始时间
    device_op_dur: float = 0.0        # Device 算子耗时 (us)

    # 关联信息
    connection_id: int = -1           # 关联标识（关键字段）
    rank_id: int = 0                  # Rank ID

    # 堆栈信息
    python_stack: List[str] = field(default_factory=list)     # Python 堆栈（前 N 帧）
    cpp_stack: List[str] = field(default_factory=list)        # C++ 堆栈（前 N 帧）
    host_stack: Optional[HostStack] = None                    # 完整 Host 堆栈

    # 特征识别
    source_type: str = "unknown"      # "eager" / "torch_compile" / "fusion_op" / "mindspeed" / etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "torch_op_name": self.torch_op_name,
            "torch_op_ts": self.torch_op_ts,
            "torch_op_dur": self.torch_op_dur,
            "acl_api_name": self.acl_api_name,
            "acl_api_ts": self.acl_api_ts,
            "acl_api_dur": self.acl_api_dur,
            "device_op_name": self.device_op_name,
            "device_op_ts": self.device_op_ts,
            "device_op_dur": self.device_op_dur,
            "connection_id": self.connection_id,
            "rank_id": self.rank_id,
            "python_stack": self.python_stack,
            "cpp_stack": self.cpp_stack,
            "source_type": self.source_type,
        }

    def get_latency_breakdown(self) -> Dict[str, float]:
        """获取延迟分解"""
        return {
            "torch_overhead": self.torch_op_dur - self.acl_api_dur if self.acl_api_dur > 0 else self.torch_op_dur,
            "acl_overhead": self.acl_api_dur - self.device_op_dur if self.device_op_dur > 0 else 0,
            "device_time": self.device_op_dur,
        }


@dataclass
class CorrelationStats:
    """关联统计"""
    total_chains: int = 0
    by_source_type: Dict[str, int] = field(default_factory=dict)     # {source_type: count}
    by_operator: Dict[str, int] = field(default_factory=dict)        # {op_name: count}

    # 关键发现
    eager_ops: List[str] = field(default_factory=list)               # Eager 模式算子
    compile_ops: List[str] = field(default_factory=list)             # torch.compile 算子
    fusion_ops: List[str] = field(default_factory=list)              # 融合算子
    mindspeed_ops: List[str] = field(default_factory=list)           # mindspeed 算子

    # 对比分析用
    operator_source_map: Dict[str, str] = field(default_factory=dict)  # {op_name: source_type}

    # 高频堆栈模式
    top_stack_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chains": self.total_chains,
            "by_source_type": self.by_source_type,
            "by_operator": dict(sorted(self.by_operator.items(), key=lambda x: x[1], reverse=True)[:50]),
            "eager_ops": self.eager_ops[:20],
            "compile_ops": self.compile_ops[:20],
            "fusion_ops": self.fusion_ops[:20],
            "mindspeed_ops": self.mindspeed_ops[:20],
            "top_stack_patterns": self.top_stack_patterns[:10],
        }


@dataclass
class SourceAnalysisResult:
    """算子来源分析结果"""
    total_chains: int = 0
    by_source_type: Dict[str, int] = field(default_factory=dict)
    top_source_operators: Dict[str, List[str]] = field(default_factory=dict)
    stack_patterns: List[str] = field(default_factory=list)

    # 潜在问题识别
    potential_issues: List[str] = field(default_factory=list)

    # 统计信息
    stats: Optional[CorrelationStats] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chains": self.total_chains,
            "by_source_type": self.by_source_type,
            "top_source_operators": self.top_source_operators,
            "stack_patterns": self.stack_patterns,
            "potential_issues": self.potential_issues,
        }

    def to_prompt_text(self) -> str:
        """转为适合 LLM 分析的文本"""
        lines = ["## 算子来源分析", ""]

        if self.by_source_type:
            lines.append("### 来源分布")
            for source, count in sorted(self.by_source_type.items(), key=lambda x: x[1], reverse=True):
                pct = count / self.total_chains * 100 if self.total_chains > 0 else 0
                lines.append(f"- {source}: {count} ({pct:.1f}%)")
            lines.append("")

        if self.top_source_operators:
            lines.append("### 各来源 Top 算子")
            for source, ops in self.top_source_operators.items():
                if ops:
                    lines.append(f"**{source}**: {', '.join(ops[:10])}")
            lines.append("")

        if self.potential_issues:
            lines.append("### 潜在问题")
            for issue in self.potential_issues:
                lines.append(f"- {issue}")
            lines.append("")

        return "\n".join(lines)


# 堆栈模式定义
STACK_PATTERNS = {
    # torch.compile 图模式
    "torch_compile": {
        "patterns": [
            "CompiledFunctionBackward",
            "CompiledFunction",
            "CUDAGraph",
            "torch._dynamo",
            "torch._inductor",
            "TorchDynamo",
            "AOTAutograd",
        ],
        "label": "torch.compile 图模式",
    },

    # 融合算子（昇腾原生）
    "fusion_op": {
        "patterns": [
            "NPUGroupedLinearGMM",
            "NPUGroupedMatmul",
            "FlashAttention",
            "FusedMatmul",
            "aclnnGroupedMatmul",
            "FusedScaleMaskSoftmax",
            "FusedLayerNorm",
            "FusedRMSNorm",
            "npu_scaled_masked_softmax",
            "npu_fusion",
        ],
        "label": "融合算子",
    },

    # eager 模式
    "eager": {
        "patterns": [
            "aten::",
            "torch.ops",
        ],
        "excludes": ["CompiledFunction", "torch._dynamo", "torch._inductor"],
        "label": "Eager 模式",
    },

    # mindspeed 相关
    "mindspeed": {
        "patterns": [
            "mindspeed.",
            "mindspeed_",
            "megatron.",
            "Megatron",
            "transformer_module",
            "ParallelMLP",
            "ParallelAttention",
            "ColumnParallelLinear",
            "RowParallelLinear",
            "VocabParallelEmbedding",
        ],
        "label": "MindSpeed/Megatron",
    },

    # torch-ascend / CANN 相关
    "torch_ascend": {
        "patterns": [
            "torch_npu",
            "aten_npu",
            "aclnn",
            "AscendCL",
            "aclOp",
            "OpRunner",
            "AclOpExecutor",
            "cann",
        ],
        "label": "Torch-Ascend/CANN",
    },

    # 分布式通信相关
    "distributed": {
        "patterns": [
            "torch.distributed",
            "ProcessGroup",
            "HCCL",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "broadcast",
            "ProcessGroupHCCL",
        ],
        "label": "分布式通信",
    },

    # 优化器相关
    "optimizer": {
        "patterns": [
            "Optimizer.step",
            "AdamW",
            "Adam",
            "LAMB",
            "FusedAdam",
            "FusedLAMB",
            "_fused_adam",
            "_multi_tensor_adam",
        ],
        "label": "优化器",
    },
}


def get_pattern_label(source_type: str) -> str:
    """获取来源类型的标签"""
    if source_type in STACK_PATTERNS:
        return STACK_PATTERNS[source_type]["label"]
    return source_type
