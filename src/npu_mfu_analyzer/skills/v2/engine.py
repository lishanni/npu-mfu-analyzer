"""
Skills v2 执行引擎

负责技能的执行、链式调用和结果聚合。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
import time
import asyncio

from .base import (
    BaseSkill,
    SkillType,
    SkillContext,
    SkillResult,
    SkillMetadata,
)
from .registry import SkillRegistry, get_registry

logger = logging.getLogger(__name__)


class ChainStatus(Enum):
    """技能链执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # 部分成功
    SKIPPED = "skipped"


@dataclass
class ChainStep:
    """技能链步骤"""
    skill_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[[SkillContext], bool]] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    result: Optional[SkillResult] = None

    def check_condition(self, context: SkillContext) -> bool:
        """检查执行条件"""
        if self.condition is None:
            return True
        try:
            return self.condition(context)
        except Exception as e:
            logger.warning(f"条件检查失败: {e}")
            return False


@dataclass
class ChainResult:
    """技能链执行结果"""
    chain_name: str
    status: ChainStatus
    steps: List[ChainStep] = field(default_factory=list)
    results: Dict[str, SkillResult] = field(default_factory=dict)
    total_time_ms: float = 0
    error: Optional[str] = None

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results.values() if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results.values() if not r.success)

    def to_markdown(self) -> str:
        """转换为 Markdown"""
        lines = [
            f"# {self.chain_name} 执行报告",
            "",
            f"**状态**: {self.status.value}",
            f"**成功**: {self.success_count} / {len(self.results)}",
            f"**耗时**: {self.total_time_ms:.2f}ms",
            "",
        ]

        for step in self.steps:
            if step.result:
                status = "✅" if step.result.success else "❌"
                lines.append(f"## {status} {step.skill_name}")
                lines.append(step.result.to_markdown())
                lines.append("")

        return "\n".join(lines)


@dataclass
class SkillChain:
    """
    技能链定义

    支持链式调用和条件执行
    """

    name: str
    description: str = ""
    steps: List[ChainStep] = field(default_factory=list)

    def add_step(
        self,
        skill_name: str,
        inputs: Dict[str, Any] = None,
        condition: Callable[[SkillContext], bool] = None,
        on_success: str = None,
        on_failure: str = None,
    ) -> "SkillChain":
        """添加步骤（支持链式调用）"""
        self.steps.append(ChainStep(
            skill_name=skill_name,
            inputs=inputs or {},
            condition=condition,
            on_success=on_success,
            on_failure=on_failure,
        ))
        return self

    def then(self, skill_name: str, **kwargs) -> "SkillChain":
        """添加下一步（语法糖）"""
        return self.add_step(skill_name, **kwargs)


class SkillEngine:
    """
    技能执行引擎

    负责：
    1. 单个技能的执行
    2. 技能链的执行
    3. 结果聚合
    """

    def __init__(self, registry: SkillRegistry = None):
        self.registry = registry or get_registry()
        self._chains: Dict[str, SkillChain] = {}
        self._execution_history: List[SkillResult] = []
        self._hooks: Dict[str, List[Callable]] = {
            "before_execute": [],
            "after_execute": [],
            "on_error": [],
        }

    def register_chain(self, chain: SkillChain) -> None:
        """注册技能链"""
        self._chains[chain.name] = chain
        logger.debug(f"注册技能链: {chain.name}")

    def build_chain(self, name: str, description: str = "") -> SkillChain:
        """
        构建技能链（Builder 模式）

        用法:
            engine.build_chain("mfu_diagnosis") \\
                .then("compute_mfu") \\
                .then("analyze_timeline")
        """
        chain = SkillChain(name=name, description=description)
        self.register_chain(chain)
        return chain

    def add_hook(self, event: str, callback: Callable) -> None:
        """
        添加钩子

        支持: before_execute, after_execute, on_error
        """
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _run_hooks(self, event: str, *args, **kwargs) -> None:
        """运行钩子"""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"钩子 {event} 执行失败: {e}")

    def execute_skill(
        self,
        skill_name: str,
        context: SkillContext,
    ) -> SkillResult:
        """
        执行单个技能

        Args:
            skill_name: 技能名称
            context: 执行上下文

        Returns:
            SkillResult: 执行结果
        """
        skill = self.registry.get(skill_name)

        if skill is None:
            return SkillResult(
                skill_name=skill_name,
                skill_type=SkillType.COMPUTE,
                success=False,
                error=f"技能 '{skill_name}' 不存在",
            )

        # 运行前置钩子
        self._run_hooks("before_execute", skill, context)

        try:
            result = skill.run(context)
            self._execution_history.append(result)

            logger.info(
                f"执行技能 {skill_name}: "
                f"{'成功' if result.success else '失败'} "
                f"({result.execution_time_ms:.2f}ms)"
            )

            # 运行后置钩子
            self._run_hooks("after_execute", skill, context, result)

            return result

        except Exception as e:
            logger.exception(f"技能 {skill_name} 执行异常")

            result = SkillResult(
                skill_name=skill_name,
                skill_type=skillType.COMPUTE,
                success=False,
                error=str(e),
            )

            self._run_hooks("on_error", skill, context, e)
            return result

    def execute_skills(
        self,
        skill_names: List[str],
        context: SkillContext,
    ) -> Dict[str, SkillResult]:
        """
        执行多个技能（自动处理依赖）

        Args:
            skill_names: 技能名称列表
            context: 执行上下文

        Returns:
            Dict[str, SkillResult]: 所有技能的执行结果
        """
        # 验证依赖
        missing = self.registry.validate_dependencies(skill_names)
        if missing:
            logger.warning(f"缺失依赖: {missing}")

        # 计算执行顺序
        order = self.registry.get_execution_order(skill_names)

        results: Dict[str, SkillResult] = {}

        for skill_name in order:
            # 更新上下文中的前序结果
            context.previous_results = results

            # 执行技能
            result = self.execute_skill(skill_name, context)
            results[skill_name] = result

        return results

    def execute_chain(
        self,
        chain_name: str,
        context: SkillContext,
    ) -> ChainResult:
        """
        执行技能链

        Args:
            chain_name: 技能链名称
            context: 执行上下文

        Returns:
            ChainResult: 链执行结果
        """
        if chain_name not in self._chains:
            return ChainResult(
                chain_name=chain_name,
                status=ChainStatus.FAILED,
                error=f"技能链 '{chain_name}' 不存在",
            )

        chain = self._chains[chain_name]
        start_time = time.time()
        results: Dict[str, SkillResult] = {}
        executed_steps: List[ChainStep] = []

        for step in chain.steps:
            # 检查条件
            if not step.check_condition(context):
                logger.debug(f"跳过步骤 {step.skill_name}: 条件不满足")
                continue

            # 更新上下文
            context.previous_results = results
            context.user_inputs.update(step.inputs)

            # 执行技能
            result = self.execute_skill(step.skill_name, context)
            step.result = result
            executed_steps.append(step)
            results[step.skill_name] = result

        total_time = (time.time() - start_time) * 1000

        # 确定状态
        success_count = sum(1 for r in results.values() if r.success)
        if success_count == len(results):
            status = ChainStatus.SUCCESS
        elif success_count == 0:
            status = ChainStatus.FAILED
        else:
            status = ChainStatus.PARTIAL

        return ChainResult(
            chain_name=chain_name,
            status=status,
            steps=executed_steps,
            results=results,
            total_time_ms=total_time,
        )

    async def execute_skill_async(
        self,
        skill_name: str,
        context: SkillContext,
    ) -> SkillResult:
        """异步执行技能"""
        # 在线程池中执行同步技能
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.execute_skill,
            skill_name,
            context,
        )

    async def execute_skills_async(
        self,
        skill_names: List[str],
        context: SkillContext,
        parallel: bool = False,
    ) -> Dict[str, SkillResult]:
        """
        异步执行多个技能

        Args:
            skill_names: 技能名称列表
            context: 执行上下文
            parallel: 是否并行执行（不考虑依赖）
        """
        if parallel:
            # 并行执行（不保证依赖顺序）
            tasks = [
                self.execute_skill_async(name, context)
                for name in skill_names
            ]
            results = await asyncio.gather(*tasks)
            return dict(zip(skill_names, results))
        else:
            # 串行执行（处理依赖）
            order = self.registry.get_execution_order(skill_names)
            results = {}
            for name in order:
                context.previous_results = results
                result = await self.execute_skill_async(name, context)
                results[name] = result
            return results

    def get_execution_history(self, limit: int = 10) -> List[SkillResult]:
        """获取执行历史"""
        return self._execution_history[-limit:]

    def clear_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()

    def get_available_skills_prompt(self) -> str:
        """生成可用技能列表（供 LLM 参考）"""
        return self.registry.get_skill_catalog()


# 预定义的技能链
def create_builtin_chains(engine: SkillEngine) -> None:
    """创建内置技能链"""

    # MFU 诊断链
    engine.build_chain(
        "mfu_diagnosis",
        "MFU 性能诊断链"
    ).then("compute_mfu") \
     .then("analyze_timeline") \
     .then("analyze_operators")

    # 通信瓶颈诊断链
    engine.build_chain(
        "communication_diagnosis",
        "通信瓶颈诊断链"
    ).then("analyze_communication") \
     .then("analyze_comm_matrix") \
     .then("detect_slow_ranks")

    # 根因分析链
    engine.build_chain(
        "root_cause_analysis",
        "根因分析链"
    ).then("compute_mfu") \
     .then("classify_operators") \
     .then("diagnose_root_cause")


def get_engine() -> SkillEngine:
    """获取全局技能引擎实例"""
    engine = SkillEngine()
    create_builtin_chains(engine)
    return engine