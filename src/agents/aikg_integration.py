"""
AIKG (Auto Kernel Generator) 集成模块

将检测到的融合机会转换为 AIKG 请求，并管理融合算子的生成流程。
"""

import logging
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import subprocess
import hashlib

from src.agents.fusion_rules import FusionOpportunity, ASCEND_FUSED_OPERATORS
from src.data_loader.aic_metrics import (
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    BOTTLENECK_PIPELINE,
    BOTTLENECK_BALANCED,
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
)

logger = logging.getLogger(__name__)


class AIKGBackend(Enum):
    """AIKG 支持的后端"""
    ASCEND = "ascend"
    CUDA = "cuda"
    CPU = "cpu"


class GenerationStatus(Enum):
    """生成状态"""
    PENDING = "pending"
    GENERATING = "generating"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AIKGRequest:
    """
    AIKG 生成请求

    将 FusionOpportunity 转换为 AIKG 可处理的格式
    """
    fusion_name: str                    # 融合算子名称
    fusion_description: str             # 融合描述
    operator_sequence: List[str]        # 算子序列名称
    operator_pattern: str               # 算子模式（正则描述）

    # 算子信息
    input_shapes: List[List[int]]       # 输入形状（示例）
    output_shapes: List[List[int]]      # 输出形状（示例）
    data_types: List[str]               # 数据类型（FP16, BF16, FP32）

    # 性能目标
    target_speedup: float               # 目标加速比
    target_backend: AIKGBackend         # 目标后端

    # 元数据
    original_opportunity_id: str        # 原始融合机会 ID
    complexity: str                     # 实现复杂度
    estimated_memory_saving: float      # 预估内存节省

    # Triton 提示（如果已有）
    triton_hint: Optional[str] = None   # Triton 实现提示

    # === 硬件指标字段（从 AIC metrics 提取） ===
    # 算术单元利用率约束
    cube_utilization: Optional[float] = None  # Cube 利用率 (0-100)
    vector_utilization: Optional[float] = None  # Vector 利用率 (0-100)

    # 内存约束
    l2_cache_hit_rate: Optional[float] = None  # L2 缓存命中率 (0-100)
    ub_usage_limit: Optional[float] = None  # UB 使用率上限 (0-100)

    # 流水线约束
    pipeline_utilization: Optional[float] = None  # 流水线利用率 (0-100)
    stall_rate_target: Optional[float] = None  # 目标停顿率 (0-100)

    # 瓶颈类型
    bottleneck_type: Optional[str] = None  # "compute", "memory", "pipeline", "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            "fusion_name": self.fusion_name,
            "fusion_description": self.fusion_description,
            "operator_sequence": self.operator_sequence,
            "operator_pattern": self.operator_pattern,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "data_types": self.data_types,
            "target_speedup": self.target_speedup,
            "target_backend": self.target_backend.value,
            "original_opportunity_id": self.original_opportunity_id,
            "complexity": self.complexity,
            "estimated_memory_saving": self.estimated_memory_saving,
            "triton_hint": self.triton_hint,
            # 硬件指标字段
            "cube_utilization": self.cube_utilization,
            "vector_utilization": self.vector_utilization,
            "l2_cache_hit_rate": self.l2_cache_hit_rate,
            "ub_usage_limit": self.ub_usage_limit,
            "pipeline_utilization": self.pipeline_utilization,
            "stall_rate_target": self.stall_rate_target,
            "bottleneck_type": self.bottleneck_type,
        }

    def to_aikg_prompt(self) -> str:
        """
        转换为 AIKG LLM Prompt

        生成适合 AIKG 的代码生成提示词
        """
        prompt_parts = [
            f"# Fusion Operator Generation Request",
            f"",
            f"## Fusion Name",
            f"{self.fusion_name}",
            f"",
            f"## Description",
            f"{self.fusion_description}",
            f"",
            f"## Operator Sequence",
            f"The following operators should be fused into a single kernel:",
        ]

        for i, op in enumerate(self.operator_sequence, 1):
            prompt_parts.append(f"{i}. {op}")

        prompt_parts.extend([
            f"",
            f"## Input/Output Specifications",
            f"- Data Types: {', '.join(self.data_types)}",
            f"- Target Backend: {self.target_backend.value}",
        ])

        if self.input_shapes:
            prompt_parts.append(f"- Input Shapes (example): {self.input_shapes}")
        if self.output_shapes:
            prompt_parts.append(f"- Output Shapes (example): {self.output_shapes}")

        prompt_parts.extend([
            f"",
            f"## Performance Goals",
            f"- Target Speedup: {self.target_speedup:.1f}x",
            f"- Estimated Memory Saving: {self.estimated_memory_saving*100:.0f}%",
            f"- Implementation Complexity: {self.complexity}",
            f"",
        ])

        # === 硬件约束部分（新增） ===
        if any([self.cube_utilization is not None,
                self.l2_cache_hit_rate is not None,
                self.pipeline_utilization is not None]):

            prompt_parts.extend([
                f"## Hardware Constraints (Based on Profiling Data)",
                f"",
            ])

            if self.cube_utilization is not None:
                prompt_parts.extend([
                    f"### Compute Constraints",
                    f"- Current Cube Utilization: {self.cube_utilization:.1f}%",
                    f"- Goal: Improve computation density and parallelism",
                    f"",
                ])

            if self.l2_cache_hit_rate is not None:
                prompt_parts.extend([
                    f"### Memory Constraints",
                    f"- Current L2 Cache Hit Rate: {self.l2_cache_hit_rate:.1f}%",
                    f"- Goal: Optimize data access pattern for better cache locality",
                    f"",
                ])

            if self.pipeline_utilization is not None:
                prompt_parts.extend([
                    f"### Pipeline Constraints",
                    f"- Current Pipeline Utilization: {self.pipeline_utilization:.1f}%",
                    f"- Goal: Improve instruction scheduling and reduce stalls",
                    f"",
                ])

            if self.bottleneck_type:
                prompt_parts.extend([
                    f"### Optimization Priority",
                    f"- Primary Bottleneck: **{self.bottleneck_type.upper()}**",
                    f"- Optimization should focus on addressing {self.bottleneck_type} constraints",
                    f"",
                ])

        prompt_parts.extend([
            f"## Requirements",
            f"1. Generate optimized Triton code for {self.target_backend.value} backend",
            f"2. Include type hints and comprehensive comments",
            f"3. Handle edge cases (empty inputs, size mismatches)",
            f"4. Optimize for memory coalescing and bank conflicts",
            f"5. Provide build script for compilation",
        ])

        if self.triton_hint:
            prompt_parts.extend([
                f"",
                f"## Implementation Hint",
                f"{self.triton_hint}",
            ])

        return "\n".join(prompt_parts)


@dataclass
class GeneratedKernel:
    """
    AIKG 生成的融合算子

    包含生成的代码、编译脚本和元数据
    """
    kernel_name: str
    request_id: str                     # 对应的 AIKGRequest ID

    # 生成内容
    triton_code: Optional[str] = None   # Triton 源代码
    build_script: Optional[str] = None  # 编译脚本
    benchmark_code: Optional[str] = None # 性能测试代码

    # 生成状态
    status: GenerationStatus = GenerationStatus.PENDING
    error_message: Optional[str] = None

    # 性能估算
    estimated_speedup: float = 1.0
    actual_speedup: Optional[float] = None  # 实测加速比

    # 文件路径（生成后）
    triton_file: Optional[Path] = None
    build_file: Optional[Path] = None
    benchmark_file: Optional[Path] = None

    # 元数据
    generation_time_ms: float = 0
    llm_model: Optional[str] = None      # 使用的 LLM 模型
    generation_params: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        """生成摘要信息"""
        return {
            "kernel_name": self.kernel_name,
            "request_id": self.request_id,
            "status": self.status.value,
            "estimated_speedup": self.estimated_speedup,
            "actual_speedup": self.actual_speedup,
            "has_triton_code": self.triton_code is not None,
            "has_build_script": self.build_script is not None,
            "files": {
                "triton": str(self.triton_file) if self.triton_file else None,
                "build": str(self.build_file) if self.build_file else None,
                "benchmark": str(self.benchmark_file) if self.benchmark_file else None,
            }
        }


class AIKGRequestConverter:
    """
    将 FusionOpportunity 转换为 AIKGRequest

    转换规则：
    1. 昇腾已有算子 (ascend_op 存在) → 跳过或标记为 SKIPPED
    2. 需要融合的算子 (opportunity_type="fuse") → 生成 AIKG 请求
    3. 自定义算子 (opportunity_type="custom") → 生成 AIKG 请求
    """

    def __init__(
        self,
        min_speedup_threshold: float = 1.05,
        max_complexity: str = "高",
        skip_native_ops: bool = True
    ):
        """
        Args:
            min_speedup_threshold: 最小加速比阈值，低于此值不生成
            max_complexity: 最大复杂度（"低"、"中等"、"高"）
            skip_native_ops: 是否跳过昇腾已有算子
        """
        self.min_speedup_threshold = min_speedup_threshold
        self.max_complexity = max_complexity
        self.skip_native_ops = skip_native_ops

        # 复杂度等级
        self._complexity_rank = {"低": 1, "中等": 2, "高": 3}

    def convert_opportunities(
        self,
        opportunities: List[FusionOpportunity]
    ) -> List[AIKGRequest]:
        """
        将融合机会列表转换为 AIKG 请求列表

        Args:
            opportunities: 融合机会列表

        Returns:
            AIKG 请求列表
        """
        requests = []

        for opp in opportunities:
            request = self._convert_single(opp)
            if request:
                requests.append(request)

        logger.info(f"Converted {len(requests)} AIKG requests from {len(opportunities)} opportunities")
        return requests

    def _convert_single(
        self,
        opportunity: FusionOpportunity
    ) -> Optional[AIKGRequest]:
        """转换单个融合机会"""
        # 1. 检查是否应该跳过
        if self._should_skip(opportunity):
            return None

        # 2. 生成唯一 ID
        opp_id = self._generate_opportunity_id(opportunity)

        # 3. 提取算子序列
        operator_names = [op.get("name", "unknown") for op in opportunity.current_ops]

        # 4. 生成输入/输出形状（使用示例形状）
        input_shapes, output_shapes = self._extract_shapes(opportunity)

        # 5. 确定数据类型
        data_types = self._infer_data_types(opportunity)

        # 6. 查找 Triton 示例（如果有）
        triton_hint = self._find_triton_hint(opportunity)

        return AIKGRequest(
            fusion_name=opportunity.name,
            fusion_description=opportunity.description,
            operator_sequence=operator_names,
            operator_pattern=self._build_operator_pattern(operator_names),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            data_types=data_types,
            target_speedup=opportunity.estimated_speedup,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id=opp_id,
            complexity=opportunity.complexity,
            estimated_memory_saving=opportunity.memory_saving,
            triton_hint=triton_hint,
        )

    def _should_skip(self, opportunity: FusionOpportunity) -> bool:
        """判断是否应该跳过此融合机会"""
        # 检查加速比
        if opportunity.end_to_end_speedup < self.min_speedup_threshold:
            logger.debug(
                f"Skipping {opportunity.name}: "
                f"speedup {opportunity.end_to_end_speedup:.2f} < threshold {self.min_speedup_threshold}"
            )
            return True

        # 检查复杂度
        opp_complexity_rank = self._complexity_rank.get(
            opportunity.complexity, 2
        )
        max_complexity_rank = self._complexity_rank.get(
            self.max_complexity, 3
        )
        if opp_complexity_rank > max_complexity_rank:
            logger.debug(
                f"Skipping {opportunity.name}: "
                f"complexity {opportunity.complexity} > max {self.max_complexity}"
            )
            return True

        # 检查是否是昇腾已有算子
        if self.skip_native_ops and opportunity.ascend_op:
            logger.debug(
                f"Skipping {opportunity.name}: "
                f"native Ascend operator exists ({opportunity.ascend_op})"
            )
            return True

        return False

    def _generate_opportunity_id(self, opportunity: FusionOpportunity) -> str:
        """生成融合机会的唯一 ID"""
        content = f"{opportunity.name}_{opportunity.opportunity_type}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_shapes(
        self,
        opportunity: FusionOpportunity
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """从融合机会中提取形状信息"""
        # 尝试从 current_ops 中获取形状
        input_shapes = []
        output_shapes = []

        for op in opportunity.current_ops[:2]:  # 只看前两个算子
            if "input_shapes" in op:
                shapes_str = op["input_shapes"]
                if shapes_str:
                    # 简单解析：假设格式为 "[M,K];[K,N]"
                    try:
                        shapes = self._parse_shape_string(shapes_str)
                        input_shapes.extend(shapes)
                    except Exception:
                        pass

            if "output_shapes" in op:
                shapes_str = op["output_shapes"]
                if shapes_str:
                    try:
                        shapes = self._parse_shape_string(shapes_str)
                        output_shapes.extend(shapes)
                    except Exception:
                        pass

        # 如果没有找到，使用默认形状
        if not input_shapes:
            input_shapes = [[1024, 1024], [1024, 1024]]  # 默认 MatMul 形状
        if not output_shapes:
            output_shapes = [[1024, 1024]]

        return input_shapes, output_shapes

    def _parse_shape_string(self, shapes_str: str) -> List[List[int]]:
        """解析形状字符串"""
        shapes = []
        for part in shapes_str.split(";"):
            part = part.strip()
            if part:
                # 移除括号，分割数字
                part = part.strip("[]()")
                dims = [int(d.strip()) for d in part.split(",") if d.strip().isdigit()]
                if dims:
                    shapes.append(dims)
        return shapes

    def _infer_data_types(self, opportunity: FusionOpportunity) -> List[str]:
        """推断数据类型"""
        # 从算子名称推断
        for op in opportunity.current_ops:
            name = op.get("name", "").lower()
            if "fp16" in name or "float16" in name:
                return ["FP16"]
            if "bf16" in name or "bfloat16" in name:
                return ["BF16"]
            if "fp32" in name or "float32" in name:
                return ["FP32"]

        # 默认返回 FP16
        return ["FP16"]

    def _find_triton_hint(self, opportunity: FusionOpportunity) -> Optional[str]:
        """查找 Triton 实现提示"""
        # 从 fusion_patterns 中查找对应的 Triton 示例
        from src.agents.fusion_rules import FUSION_PATTERNS

        for pattern in FUSION_PATTERNS:
            if pattern.name in opportunity.name or opportunity.name in pattern.name:
                if pattern.example_code:
                    return pattern.example_code

        return None

    def _build_operator_pattern(self, operator_names: List[str]) -> str:
        """构建算子模式描述"""
        # 简化：用 -> 连接算子
        return " -> ".join(operator_names)

    # ========== AIC Metrics 支持方法 ==========

    def convert_opportunities_with_aic_metrics(
        self,
        opportunities: List[FusionOpportunity],
        aic_metrics_dict: Dict[str, Any]
    ) -> List["AIKGRequest"]:
        """
        将融合机会转换为 AIKG 请求（包含 AIC metrics 硬件约束）

        Args:
            opportunities: 融合机会列表
            aic_metrics_dict: AIC 指标字典 (op_name -> AICMetrics)

        Returns:
            AIKG 请求列表（包含硬件约束）
        """
        requests = []

        for opp in opportunities:
            request = self._convert_single_with_aic(opp, aic_metrics_dict)
            if request:
                requests.append(request)

        logger.info(
            f"Converted {len(requests)} AIKG requests with AIC metrics "
            f"from {len(opportunities)} opportunities"
        )
        return requests

    def _convert_single_with_aic(
        self,
        opportunity: FusionOpportunity,
        aic_metrics_dict: Dict[str, Any]
    ) -> Optional["AIKGRequest"]:
        """转换单个融合机会（包含 AIC metrics）"""
        # 1. 检查是否应该跳过
        if self._should_skip(opportunity):
            return None

        # 2. 生成基础请求
        base_request = self._convert_single(opportunity)
        if not base_request:
            return None

        # 3. 提取 AIC metrics 硬件约束
        aic_constraints = self._extract_aic_constraints(
            opportunity, aic_metrics_dict
        )

        # 4. 更新请求对象
        base_request.cube_utilization = aic_constraints.get("cube_utilization")
        base_request.vector_utilization = aic_constraints.get("vector_utilization")
        base_request.l2_cache_hit_rate = aic_constraints.get("l2_cache_hit_rate")
        base_request.ub_usage_limit = aic_constraints.get("ub_usage_limit")
        base_request.pipeline_utilization = aic_constraints.get("pipeline_utilization")
        base_request.stall_rate_target = aic_constraints.get("stall_rate_target")
        base_request.bottleneck_type = aic_constraints.get("bottleneck_type")

        return base_request

    def _extract_aic_constraints(
        self,
        opportunity: FusionOpportunity,
        aic_metrics_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        从融合机会中提取 AIC metrics 硬件约束

        Args:
            opportunity: 融合机会
            aic_metrics_dict: AIC 指标字典

        Returns:
            硬件约束字典
        """
        constraints = {}

        # 遍历涉及的所有算子，提取最差的指标作为约束
        cube_utils = []
        l2_hit_rates = []
        pipe_utils = []

        for op in opportunity.current_ops:
            op_name = op.get("name", "")

            # 查找匹配的 AIC metrics
            metrics = None
            if op_name in aic_metrics_dict:
                metrics = aic_metrics_dict[op_name]
            else:
                # 尝试模糊匹配
                for key, m in aic_metrics_dict.items():
                    if op_name in key or key in op_name:
                        metrics = m
                        break

            if not metrics:
                continue

            # 提取指标
            if hasattr(metrics, 'arithmetic') and metrics.arithmetic:
                cube_utils.append(metrics.arithmetic.cube_utilization)

            if hasattr(metrics, 'memory') and metrics.memory:
                l2_hit_rates.append(metrics.memory.l2_cache_hit_rate)

            if hasattr(metrics, 'pipeline') and metrics.pipeline:
                pipe_utils.append(metrics.pipeline.pipe_utilization)

        # 计算约束值（使用最差值）
        if cube_utils:
            constraints["cube_utilization"] = min(cube_utils)

        if l2_hit_rates:
            constraints["l2_cache_hit_rate"] = min(l2_hit_rates)

        if pipe_utils:
            constraints["pipeline_utilization"] = min(pipe_utils)

        # 判断瓶颈类型
        if cube_utils and l2_hit_rates:
            avg_cube = sum(cube_utils) / len(cube_utils)
            avg_l2 = sum(l2_hit_rates) / len(l2_hit_rates)

            if avg_cube < CRITICAL_THRESHOLD if "CRITICAL_THRESHOLD" in globals() else avg_cube < 40:
                constraints["bottleneck_type"] = BOTTLENECK_COMPUTE if "BOTTLENECK_COMPUTE" in globals() else "compute"
            elif avg_l2 < HIGH_THRESHOLD if "HIGH_THRESHOLD" in globals() else avg_l2 < 60:
                constraints["bottleneck_type"] = BOTTLENECK_MEMORY if "BOTTLENECK_MEMORY" in globals() else "memory"
            elif pipe_utils and sum(pipe_utils) / len(pipe_utils) < MEDIUM_THRESHOLD if "MEDIUM_THRESHOLD" in globals() else 60:
                constraints["bottleneck_type"] = BOTTLENECK_PIPELINE if "BOTTLENECK_PIPELINE" in globals() else "pipeline"
            else:
                constraints["bottleneck_type"] = BOTTLENECK_BALANCED if "BOTTLENECK_BALANCED" in globals() else "balanced"

        return constraints


class AIKGKernelClient:
    """
    AIKG 内核生成客户端

    负责与 AIKG 服务通信，管理算子生成流程。
    支持：
    - 本地 LLM 调用（通过配置的 LLM 接口）
    - 远程 AIKG 服务调用
    - 模拟生成（用于测试）
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        llm_client: Optional[Any] = None,  # LLMInterface
        timeout: int = 300,
        max_concurrent: int = 3
    ):
        """
        Args:
            service_url: AIKG 服务 URL（如果使用远程服务）
            llm_client: LLM 客户端（如果使用本地 LLM）
            timeout: 生成超时时间（秒）
            max_concurrent: 最大并发生成数
        """
        self.service_url = service_url
        self._llm_client = llm_client
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        # 统计信息
        self._stats = {
            "total_requests": 0,
            "success_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
        }

    async def generate_kernels(
        self,
        requests: List[AIKGRequest],
        output_dir: Optional[Path] = None
    ) -> List[GeneratedKernel]:
        """
        批量生成融合算子

        Args:
            requests: AIKG 请求列表
            output_dir: 输出目录（如果提供，将保存生成的文件）

        Returns:
            生成的算子列表
        """
        self._stats["total_requests"] += len(requests)

        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # 限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def generate_with_semaphore(req: AIKGRequest) -> GeneratedKernel:
            async with semaphore:
                return await self._generate_single(req, output_dir)

        # 并发生成
        kernels = await asyncio.gather(
            *[generate_with_semaphore(req) for req in requests],
            return_exceptions=True
        )

        # 处理异常
        results = []
        for i, kernel in enumerate(kernels):
            if isinstance(kernel, Exception):
                logger.error(f"Request {i} failed with exception: {kernel}")
                # 创建一个失败的内核对象
                results.append(GeneratedKernel(
                    kernel_name=f"error_{i}",
                    request_id=requests[i].original_opportunity_id if i < len(requests) else "unknown",
                    status=GenerationStatus.FAILED,
                    error_message=str(kernel),
                ))
                self._stats["failed_count"] += 1
            else:
                results.append(kernel)
                if kernel.status == GenerationStatus.SUCCESS:
                    self._stats["success_count"] += 1
                elif kernel.status == GenerationStatus.FAILED:
                    self._stats["failed_count"] += 1
                else:
                    self._stats["skipped_count"] += 1

        logger.info(
            f"AIKG generation complete: "
            f"{self._stats['success_count']} success, "
            f"{self._stats['failed_count']} failed, "
            f"{self._stats['skipped_count']} skipped"
        )

        return results

    async def _generate_single(
        self,
        request: AIKGRequest,
        output_dir: Optional[Path] = None
    ) -> GeneratedKernel:
        """
        生成单个融合算子

        实现策略：
        1. 如果配置了 service_url，调用远程 AIKG 服务
        2. 如果有 llm_client，使用本地 LLM 生成
        3. 否则，返回 SKIPPED 状态（需要用户手动配置）
        """
        kernel = GeneratedKernel(
            kernel_name=request.fusion_name,
            request_id=request.original_opportunity_id,
            estimated_speedup=request.target_speedup,
        )

        # 策略 1: 远程服务
        if self.service_url:
            return await self._generate_via_service(request, kernel, output_dir)

        # 策略 2: 本地 LLM
        if self._llm_client:
            return await self._generate_via_llm(request, kernel, output_dir)

        # 策略 3: 无配置，跳过
        logger.warning(
            f"No AIKG service or LLM configured, skipping {request.fusion_name}"
        )
        kernel.status = GenerationStatus.SKIPPED
        kernel.error_message = "No AIKG service or LLM configured"
        return kernel

    async def _generate_via_service(
        self,
        request: AIKGRequest,
        kernel: GeneratedKernel,
        output_dir: Optional[Path]
    ) -> GeneratedKernel:
        """通过远程 AIKG 服务生成"""
        import aiohttp

        kernel.status = GenerationStatus.GENERATING

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_url}/generate",
                    json=request.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        kernel.triton_code = data.get("triton_code")
                        kernel.build_script = data.get("build_script")
                        kernel.benchmark_code = data.get("benchmark_code")
                        kernel.status = GenerationStatus.SUCCESS
                        kernel.generation_params = data.get("metadata", {})

                        # 保存文件
                        if output_dir:
                            self._save_kernel_files(kernel, output_dir, request)

                    else:
                        error = await response.text()
                        kernel.status = GenerationStatus.FAILED
                        kernel.error_message = f"Service error {response.status}: {error}"

        except asyncio.TimeoutError:
            kernel.status = GenerationStatus.FAILED
            kernel.error_message = f"Generation timeout after {self.timeout}s"
        except Exception as e:
            kernel.status = GenerationStatus.FAILED
            kernel.error_message = str(e)

        return kernel

    async def _generate_via_llm(
        self,
        request: AIKGRequest,
        kernel: GeneratedKernel,
        output_dir: Optional[Path]
    ) -> GeneratedKernel:
        """通过本地 LLM 生成"""
        from src.llm.llm_interface import Message

        kernel.status = GenerationStatus.GENERATING

        try:
            # 构建提示词
            prompt = request.to_aikg_prompt()

            # 保存 DSL 到文件（在当前执行目录）
            self._save_dsl_file(request, prompt)

            # 调用 LLM
            messages = [Message(role="user", content=prompt)]
            response = await self._llm_client.complete(messages)

            # 解析响应（期望 LLM 返回代码块）
            # response 是 LLMResponse 对象，需要提取 content 字段
            response_text = response.content if hasattr(response, 'content') else str(response)
            kernel.triton_code = self._extract_code_block(response_text, "python")
            kernel.build_script = self._generate_build_script(request)
            kernel.benchmark_code = self._generate_benchmark_code(request)

            kernel.status = GenerationStatus.SUCCESS
            kernel.llm_model = getattr(self._llm_client, 'model_name', 'unknown')

            # 保存文件
            if output_dir:
                self._save_kernel_files(kernel, output_dir, request)

        except Exception as e:
            kernel.status = GenerationStatus.FAILED
            kernel.error_message = str(e)
            logger.error(f"LLM generation failed for {request.fusion_name}: {e}")

        return kernel

    def _extract_code_block(self, text: str, lang: str = "python") -> Optional[str]:
        """从文本中提取代码块"""
        import re

        # 匹配 ```python ... ```（支持任意空白字符，包括闭合前的空格）
        # 使用 [\s\S]*? 而不是 .*? 来匹配包括换行在内的所有字符
        # 闭合 ``` 前允许有空白字符
        pattern = rf"```{lang}\s*\n([\s\S]*?)\n\s*```"
        match = re.search(pattern, text)
        if match:
            code = match.group(1).strip()
            # 去除每行前导空格（unindent）
            lines = code.split("\n")
            # 找到最小缩进
            min_indent = float("inf")
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # 去除最小缩进
            if min_indent < float("inf"):
                lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
            return "\n".join(lines).strip()

        # 如果没有找到代码块，尝试返回整个文本（如果看起来像代码）
        if "def " in text or "import " in text or "@" in text:
            return text.strip()

        return None

    def _generate_build_script(self, request: AIKGRequest) -> str:
        """生成编译脚本"""
        return f"""#!/bin/bash
# Build script for {request.fusion_name}

set -e

KERNEL_NAME="{request.fusion_name.replace(' ', '_').replace('-', '_').lower()}"
TRITON_FILE="${{KERNEL_NAME}}.py"

echo "Building fusion kernel: $KERNEL_NAME"

# Check if triton is available
python -c "import triton" 2>/dev/null || {{
    echo "Error: Triton not installed"
    echo "Install with: pip install triton"
    exit 1
}}

# Compile the kernel
python -c "
import torch
from ${{TRITON_FILE}} import *
print('Kernel loaded successfully')
"

echo "Build complete!"
"""

    def _generate_benchmark_code(self, request: AIKGRequest) -> str:
        """生成性能测试代码"""
        return f"""#!/usr/bin/env python3
\"\"\"
Benchmark for {request.fusion_name}

Generated by npu-mfu-analyzer AIKG integration
\"\"\"

import torch
import triton
import time
from typing import Tuple

# Import the generated kernel
# TODO: Add import statement for your kernel

def benchmark_fusion(
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    repeats: int = 100
) -> Tuple[float, float]:
    \"\"\"
    Benchmark the fused kernel

    Returns:
        (mean_time_ms, std_time_ms)
    \"\"\"
    # TODO: Implement benchmark based on your fusion

    # Placeholder implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test data
    # a = torch.randn(m, k, dtype=dtype, device=device)
    # b = torch.randn(k, n, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        # TODO: Call your kernel
        pass

    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        # TODO: Call your kernel
        # output = your_kernel(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append((end - start) * 1000)  # to ms

    import numpy as np
    return float(np.mean(times)), float(np.std(times))

if __name__ == "__main__":
    mean_time, std_time = benchmark_fusion()
    print(f"Mean: {{mean_time:.3f}} ms, Std: {{std_time:.3f}} ms")
    print(f"Estimated speedup: {request.target_speedup:.1f}x")
"""

    def _save_kernel_files(
        self,
        kernel: GeneratedKernel,
        output_dir: Path,
        request: AIKGRequest
    ):
        """保存生成的文件"""
        import hashlib
        import re

        # 生成安全的文件名（移除/替换特殊字符）
        # 使用拼音或简化的英文名称会更好，这里使用简单的转义
        safe_name = request.fusion_name.lower()
        # 移除特殊字符，只保留字母、数字、下划线和连字符
        safe_name = re.sub(r'[^\w\-]', '_', safe_name, flags=re.ASCII)
        # 移除连续的下划线
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        # 如果结果为空或太短，使用哈希值
        if len(safe_name) < 3:
            safe_name = f"fusion_{hashlib.md5(request.fusion_name.encode()).hexdigest()[:8]}"

        base_path = output_dir / safe_name

        # 保存 Triton 代码
        if kernel.triton_code:
            triton_path = str(base_path) + ".py"
            Path(triton_path).write_text(kernel.triton_code, encoding="utf-8")
            kernel.triton_file = Path(triton_path)

        # 保存编译脚本
        if kernel.build_script:
            build_path = str(base_path) + ".sh"
            Path(build_path).write_text(kernel.build_script, encoding="utf-8")
            Path(build_path).chmod(0o755)  # 可执行权限
            kernel.build_file = Path(build_path)

        # 保存性能测试
        if kernel.benchmark_code:
            bench_path = str(base_path) + "_bench.py"
            Path(bench_path).write_text(kernel.benchmark_code, encoding="utf-8")
            Path(bench_path).chmod(0o755)
            kernel.benchmark_file = Path(bench_path)

    def _save_dsl_file(self, request: AIKGRequest, dsl_content: str):
        """保存 AIKG DSL 到文件（在当前执行目录）"""
        import hashlib
        import re
        from datetime import datetime

        # 生成安全的文件名
        safe_name = request.fusion_name.lower()
        safe_name = re.sub(r'[^\w\-]', '_', safe_name, flags=re.ASCII)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        if len(safe_name) < 3:
            safe_name = f"fusion_{hashlib.md5(request.fusion_name.encode()).hexdigest()[:8]}"

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aikg_dsl_{safe_name}_{timestamp}.txt"

        # 保存到当前工作目录
        dsl_path = Path.cwd() / filename
        dsl_path.write_text(dsl_content, encoding="utf-8")

        logger.info(f"AIKG DSL saved to: {dsl_path}")

    def get_stats(self) -> Dict[str, int]:
        """获取生成统计信息"""
        return self._stats.copy()


class AIKGIntegrator:
    """
    AIKG 集成器

    完整的 AIKG 集成流程：
    1. FusionOpportunity → AIKGRequest
    2. AIKGRequest → GeneratedKernel
    3. 生成的算子保存到文件
    """

    def __init__(
        self,
        converter: Optional[AIKGRequestConverter] = None,
        client: Optional[AIKGKernelClient] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            converter: 请求转换器
            client: AIKG 客户端
            output_dir: 输出目录
        """
        self.converter = converter or AIKGRequestConverter()
        self.client = client or AIKGKernelClient()
        self.output_dir = output_dir

    async def generate_from_opportunities(
        self,
        opportunities: List[FusionOpportunity]
    ) -> List[GeneratedKernel]:
        """
        从融合机会生成融合算子

        Args:
            opportunities: 融合机会列表

        Returns:
            生成的算子列表
        """
        logger.info(f"Starting AIKG generation from {len(opportunities)} opportunities")

        # 1. 转换为 AIKG 请求
        requests = self.converter.convert_opportunities(opportunities)

        if not requests:
            logger.warning("No AIKG requests generated (all opportunities filtered)")
            return []

        # 2. 生成算子
        kernels = await self.client.generate_kernels(requests, self.output_dir)

        # 3. 记录摘要
        self._log_summary(kernels)

        return kernels

    def _log_summary(self, kernels: List[GeneratedKernel]):
        """记录生成摘要"""
        status_count = {}
        for k in kernels:
            status = k.status.value
            status_count[status] = status_count.get(status, 0) + 1

        logger.info(
            f"AIKG generation summary: {status_count.get('success', 0)} success, "
            f"{status_count.get('failed', 0)} failed, "
            f"{status_count.get('skipped', 0)} skipped"
        )

        # 列出成功的算子
        success_kernels = [k for k in kernels if k.status == GenerationStatus.SUCCESS]
        if success_kernels:
            logger.info("Successfully generated kernels:")
            for k in success_kernels:
                logger.info(f"  - {k.kernel_name} (est. speedup: {k.estimated_speedup:.1f}x)")
