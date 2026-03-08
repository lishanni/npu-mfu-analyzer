"""
NPU MFU Analyzer Skill SDK

提供标准化的 Skill 开发接口，便于集成到各种 Agent 平台。
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from pathlib import Path
import logging

from .base import (
    BaseSkill,
    SkillType,
    SkillCategory,
    SkillMetadata,
    SkillContext,
    SkillResult,
)
from .registry import SkillRegistry, get_registry
from .engine import SkillEngine
from .loader import SkillLoader

logger = logging.getLogger(__name__)


class SkillExecutor(Protocol):
    """技能执行器协议"""

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有可用技能"""
        ...

    def execute(self, skill_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行指定技能"""
        ...

    def execute_chain(
        self,
        skill_names: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """执行技能链"""
        ...


@dataclass
class AnalysisOptions:
    """分析选项"""
    skills: Optional[List[str]] = None
    use_llm: bool = True
    enable_host_device_correlation: bool = True
    enable_comm_matrix: bool = True
    enable_aic_microarch: bool = True
    output_format: str = "markdown"


class NPUMFUAnalyzerSDK:
    """
    NPU MFU Analyzer SDK

    提供统一的接口，便于集成到各种 Agent 平台。

    Usage:
        # 基本使用
        sdk = NPUMFUAnalyzerSDK()
        result = sdk.analyze("/path/to/profiling")

        # 指定技能
        result = sdk.analyze(
            "/path/to/profiling",
            skills=["compute_mfu", "diagnose_performance"]
        )

        # 对比分析
        result = sdk.compare("/path/to/v1", "/path/to/v2")
    """

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 SDK

        Args:
            skills_dir: 自定义技能目录
            llm_config: LLM 配置
        """
        self.registry = SkillRegistry()
        self.engine = SkillEngine(self.registry)
        self.loader = SkillLoader(self.registry)
        self.llm_config = llm_config

        # 加载技能
        self._load_skills(skills_dir)

    def _load_skills(self, skills_dir: Optional[str]) -> None:
        """加载技能"""
        # 1. 加载内置技能
        builtin_skills_dir = Path(__file__).parent / "skills"
        if builtin_skills_dir.exists():
            self.loader.discover(builtin_skills_dir)

        # 2. 加载自定义技能
        if skills_dir:
            self.loader.discover(Path(skills_dir))

    def list_skills(self) -> List[Dict[str, Any]]:
        """
        列出所有可用技能

        Returns:
            技能元数据列表
        """
        return [skill.metadata.to_dict() for skill in self.registry.list_all()]

    def get_skill(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取技能详情

        Args:
            name: 技能名称

        Returns:
            技能元数据，不存在返回 None
        """
        meta = self.registry.get_metadata(name)
        return meta.to_dict() if meta else None

    def analyze(
        self,
        profiling_path: str,
        options: Optional[AnalysisOptions] = None,
    ) -> Dict[str, Any]:
        """
        分析 Profiling 数据

        Args:
            profiling_path: Profiling 数据路径
            options: 分析选项

        Returns:
            分析结果
        """
        options = options or AnalysisOptions()

        # 1. 加载 Profiling 数据
        context = self._load_profiling_context(profiling_path)

        # 2. 确定要执行的技能
        skill_names = options.skills or self._select_skills(context, options)

        # 3. 执行技能
        results = self.engine.execute_skills(skill_names, context)

        # 4. LLM 综合推理（可选）
        if options.use_llm and self.llm_config:
            llm_result = self._llm_reasoning(context, results)
            results["llm_reasoning"] = llm_result

        # 5. 格式化输出
        return self._format_output(results, options.output_format)

    def compare(
        self,
        path_a: str,
        path_b: str,
        options: Optional[AnalysisOptions] = None,
    ) -> Dict[str, Any]:
        """
        对比两个 Profiling 数据

        Args:
            path_a: 基准 Profiling 路径
            path_b: 对照 Profiling 路径
            options: 分析选项

        Returns:
            对比分析结果
        """
        options = options or AnalysisOptions()

        # 1. 加载两组 Profiling 数据
        context_a = self._load_profiling_context(path_a)
        context_b = self._load_profiling_context(path_b)

        # 2. 构建对比上下文
        context = context_a
        context.profiling_summary_b = context_b.profiling_summary
        context.hardware_spec_b = context_b.hardware_spec

        # 3. 计算差异
        diff_skills = [
            "compute_diff",
            "diagnose_root_cause",
        ]
        skill_names = options.skills or diff_skills

        # 4. 执行技能
        results = self.engine.execute_skills(skill_names, context)

        return self._format_output(results, options.output_format)

    def execute_skill(
        self,
        skill_name: str,
        profiling_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行单个技能

        Args:
            skill_name: 技能名称
            profiling_path: Profiling 数据路径
            **kwargs: 额外参数

        Returns:
            技能执行结果
        """
        context = self._load_profiling_context(profiling_path)
        context.user_inputs = kwargs

        result = self.engine.execute_skill(skill_name, context)
        return result.to_dict()

    def _load_profiling_context(self, profiling_path: str) -> SkillContext:
        """加载 Profiling 上下文"""
        from src.data_loader.profiling_loader import ProfilingLoader
        from src.data_loader.data_summarizer import DataSummarizer

        loader = ProfilingLoader(profiling_path)
        summarizer = DataSummarizer(loader)

        return SkillContext(
            profiling_summary=summarizer.summarize(),
            hardware_spec=loader.detect_hardware(),
            pattern=loader.detect_pattern(),
            config={"profiling_path": profiling_path},
        )

    def _select_skills(
        self,
        context: SkillContext,
        options: AnalysisOptions,
    ) -> List[str]:
        """根据上下文自动选择技能"""
        selected = []

        # 核心技能
        selected.extend([
            "compute_mfu",
            "analyze_timeline",
            "analyze_communication",
        ])

        # 条件技能
        if options.enable_host_device_correlation and context.host_device_chains:
            selected.extend([
                "classify_operators",
                "diagnose_root_cause",
            ])

        if options.enable_comm_matrix:
            selected.append("analyze_comm_matrix")

        if options.enable_aic_microarch:
            # 检查是否有 AIC metrics
            if hasattr(context.profiling_summary, "has_aic_metrics"):
                if context.profiling_summary.has_aic_metrics:
                    selected.append("analyze_aic_microarch")

        return selected

    def _llm_reasoning(
        self,
        context: SkillContext,
        results: Dict[str, SkillResult],
    ) -> SkillResult:
        """LLM 综合推理"""
        # TODO: 实现 LLM 推理
        return SkillResult(
            skill_name="llm_reasoning",
            skill_type=SkillType.REASONING,
            success=False,
            error="LLM 推理尚未实现",
        )

    def _format_output(
        self,
        results: Dict[str, SkillResult],
        format: str,
    ) -> Dict[str, Any]:
        """格式化输出"""
        output = {
            "success": all(r.success for r in results.values()),
            "results": {name: r.to_dict() for name, r in results.items()},
        }

        if format == "markdown":
            lines = ["# 分析报告\n"]
            for name, result in results.items():
                lines.append(result.to_markdown())
                lines.append("\n---\n")
            output["markdown"] = "\n".join(lines)

        return output


# 便捷函数
def analyze(profiling_path: str, **kwargs) -> Dict[str, Any]:
    """
    便捷分析函数

    Args:
        profiling_path: Profiling 数据路径
        **kwargs: 额外参数

    Returns:
        分析结果
    """
    sdk = NPUMFUAnalyzerSDK()
    return sdk.analyze(profiling_path, AnalysisOptions(**kwargs))


def compare(path_a: str, path_b: str, **kwargs) -> Dict[str, Any]:
    """
    便捷对比函数

    Args:
        path_a: 基准 Profiling 路径
        path_b: 对照 Profiling 路径
        **kwargs: 额外参数

    Returns:
        对比结果
    """
    sdk = NPUMFUAnalyzerSDK()
    return sdk.compare(path_a, path_b, AnalysisOptions(**kwargs))


def list_skills() -> List[Dict[str, Any]]:
    """列出所有可用技能"""
    sdk = NPUMFUAnalyzerSDK()
    return sdk.list_skills()