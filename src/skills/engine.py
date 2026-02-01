"""
Skill Engine

技能执行引擎，负责调用技能并管理执行流程
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import time
from enum import Enum

from .base_skill import (
    BaseSkill,
    PromptSkill,
    SkillResult,
    SkillCategory,
)
from .registry import get_registry, SkillRegistry

logger = logging.getLogger(__name__)


class ChainStatus(Enum):
    """逻辑链执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # 部分成功


@dataclass
class ChainStep:
    """逻辑链步骤"""
    skill_name: str               # 要调用的技能名称
    inputs: Dict[str, Any] = field(default_factory=dict)  # 输入参数
    condition: Optional[str] = None  # 条件表达式（可选）
    on_success: Optional[str] = None  # 成功后执行的下一步（skill_name）
    on_failure: Optional[str] = None  # 失败后执行的下一步
    result: Optional[SkillResult] = None  # 执行结果


@dataclass
class ChainResult:
    """逻辑链执行结果"""
    chain_name: str
    status: ChainStatus
    steps: List[ChainStep] = field(default_factory=list)
    total_time_ms: float = 0
    final_summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    def to_prompt_text(self) -> str:
        """转换为 LLM 可读的文本"""
        lines = [
            f"# {self.chain_name} 执行报告",
            f"状态: {self.status.value}",
            f"总耗时: {self.total_time_ms:.2f}ms",
            "",
            "## 执行步骤",
        ]
        
        for i, step in enumerate(self.steps, 1):
            status = "✓" if step.result and step.result.success else "✗"
            lines.append(f"\n### {i}. {step.skill_name} [{status}]")
            if step.result:
                lines.append(step.result.to_prompt_text())
        
        if self.final_summary:
            lines.append(f"\n## 总结\n{self.final_summary}")
        
        if self.suggestions:
            lines.append("\n## 优化建议")
            for i, s in enumerate(self.suggestions, 1):
                lines.append(f"{i}. {s}")
        
        return "\n".join(lines)


@dataclass
class LogicChain:
    """
    逻辑链定义
    
    定义一系列有序的技能调用步骤
    """
    name: str
    description: str
    steps: List[ChainStep] = field(default_factory=list)
    
    def add_step(
        self,
        skill_name: str,
        inputs: Dict[str, Any] = None,
        condition: str = None,
        on_success: str = None,
        on_failure: str = None,
    ) -> "LogicChain":
        """添加步骤（支持链式调用）"""
        self.steps.append(ChainStep(
            skill_name=skill_name,
            inputs=inputs or {},
            condition=condition,
            on_success=on_success,
            on_failure=on_failure,
        ))
        return self


class SkillEngine:
    """
    技能执行引擎
    
    负责：
    1. 单个技能的执行
    2. 逻辑链的执行
    3. 结果聚合和转换
    """
    
    def __init__(self, registry: SkillRegistry = None):
        self.registry = registry or get_registry()
        self._chains: Dict[str, LogicChain] = {}
        self._execution_history: List[SkillResult] = []
    
    def execute_skill(
        self,
        skill_name: str,
        **kwargs,
    ) -> SkillResult:
        """
        执行单个技能
        
        Args:
            skill_name: 技能名称
            **kwargs: 技能输入参数
            
        Returns:
            SkillResult: 执行结果
        """
        skill = self.registry.get_python_skill(skill_name)
        
        if skill is None:
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"技能 '{skill_name}' 不存在",
            )
        
        result = skill.run(**kwargs)
        self._execution_history.append(result)
        
        logger.info(
            f"执行技能 {skill_name}: "
            f"{'成功' if result.success else '失败'} "
            f"({result.execution_time_ms:.2f}ms)"
        )
        
        return result
    
    def render_prompt_skill(
        self,
        skill_name: str,
        **kwargs,
    ) -> Optional[str]:
        """
        渲染 Prompt 技能
        
        Args:
            skill_name: Prompt 技能名称
            **kwargs: 模板变量
            
        Returns:
            渲染后的 prompt 文本
        """
        skill = self.registry.get_prompt_skill(skill_name)
        
        if skill is None:
            logger.warning(f"Prompt 技能 '{skill_name}' 不存在")
            return None
        
        return skill.render(**kwargs)
    
    def register_chain(self, chain: LogicChain) -> None:
        """注册逻辑链"""
        self._chains[chain.name] = chain
        logger.debug(f"注册逻辑链: {chain.name}")
    
    def execute_chain(
        self,
        chain_name: str,
        context: Dict[str, Any] = None,
    ) -> ChainResult:
        """
        执行逻辑链
        
        Args:
            chain_name: 逻辑链名称
            context: 共享上下文（可在步骤间传递数据）
            
        Returns:
            ChainResult: 链执行结果
        """
        if chain_name not in self._chains:
            return ChainResult(
                chain_name=chain_name,
                status=ChainStatus.FAILED,
                final_summary=f"逻辑链 '{chain_name}' 不存在",
            )
        
        chain = self._chains[chain_name]
        context = context or {}
        
        start_time = time.time()
        executed_steps = []
        all_suggestions = []
        success_count = 0
        
        for step in chain.steps:
            # 检查条件
            if step.condition:
                try:
                    if not eval(step.condition, {"context": context}):
                        logger.debug(f"跳过步骤 {step.skill_name}: 条件不满足")
                        continue
                except Exception as e:
                    logger.warning(f"条件表达式错误: {e}")
            
            # 合并上下文和步骤输入
            inputs = {**context, **step.inputs}
            
            # 执行技能
            result = self.execute_skill(step.skill_name, **inputs)
            step.result = result
            executed_steps.append(step)
            
            if result.success:
                success_count += 1
                all_suggestions.extend(result.suggestions)
                
                # 更新上下文
                context[f"{step.skill_name}_result"] = result.data
                
                # 跳转到成功后的步骤
                if step.on_success:
                    # 找到对应步骤并添加到执行队列
                    pass  # 简化实现，暂不支持动态跳转
            else:
                # 跳转到失败后的步骤
                if step.on_failure:
                    pass  # 简化实现
        
        total_time = (time.time() - start_time) * 1000
        
        # 确定最终状态
        if success_count == len(executed_steps):
            status = ChainStatus.SUCCESS
        elif success_count == 0:
            status = ChainStatus.FAILED
        else:
            status = ChainStatus.PARTIAL
        
        return ChainResult(
            chain_name=chain_name,
            status=status,
            steps=executed_steps,
            total_time_ms=total_time,
            suggestions=list(set(all_suggestions)),  # 去重
        )
    
    def build_chain(self, name: str, description: str = "") -> LogicChain:
        """
        构建逻辑链（Builder 模式）
        
        用法:
            engine.build_chain("mfu_diagnosis") \\
                .add_step("calculate_mfu", inputs={...}) \\
                .add_step("check_bandwidth", inputs={...})
        """
        chain = LogicChain(name=name, description=description)
        self.register_chain(chain)
        return chain
    
    def get_skill_for_pattern(
        self,
        pattern: str,
    ) -> List[str]:
        """
        根据模式匹配推荐技能
        
        Args:
            pattern: 问题模式描述
            
        Returns:
            推荐的技能名称列表
        """
        # 关键词到技能的映射
        pattern_skill_map = {
            "mfu": ["calculate_mfu", "check_compute_efficiency"],
            "带宽": ["check_bandwidth_efficiency", "analyze_collective_ops"],
            "bandwidth": ["check_bandwidth_efficiency", "analyze_collective_ops"],
            "通信": ["analyze_collective_ops", "check_overlap_ratio"],
            "communication": ["analyze_collective_ops", "check_overlap_ratio"],
            "内存": ["diagnose_memory_usage", "check_memory_fragmentation"],
            "memory": ["diagnose_memory_usage", "check_memory_fragmentation"],
            "慢卡": ["detect_slow_rank", "analyze_cross_rank_jitter"],
            "slow": ["detect_slow_rank", "analyze_cross_rank_jitter"],
            "抖动": ["detect_compute_jitter", "detect_comm_jitter"],
            "jitter": ["detect_compute_jitter", "detect_comm_jitter"],
            "overlap": ["check_overlap_ratio", "verify_overlap_strategy"],
            "掩盖": ["check_overlap_ratio", "verify_overlap_strategy"],
        }
        
        pattern_lower = pattern.lower()
        recommended = []
        
        for keyword, skills in pattern_skill_map.items():
            if keyword in pattern_lower:
                for skill_name in skills:
                    if self.registry.get_python_skill(skill_name):
                        recommended.append(skill_name)
        
        return list(set(recommended))
    
    def get_execution_history(
        self,
        limit: int = 10,
    ) -> List[SkillResult]:
        """获取最近的执行历史"""
        return self._execution_history[-limit:]
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()
    
    def get_available_skills_prompt(self) -> str:
        """
        生成可用技能列表（供 LLM 参考）
        
        Returns:
            Markdown 格式的技能列表
        """
        return self.registry.get_skill_catalog()


# 预定义的逻辑链
def create_builtin_chains(engine: SkillEngine) -> None:
    """创建内置逻辑链"""
    
    # MFU 诊断链
    engine.build_chain(
        "mfu_diagnosis",
        "MFU 性能诊断链：计算 MFU → 检查计算效率 → 检查带宽 → 生成建议"
    ).add_step(
        "calculate_mfu",
        inputs={},
    ).add_step(
        "check_compute_efficiency",
        inputs={},
    ).add_step(
        "check_bandwidth_efficiency",
        inputs={},
    )
    
    # 通信瓶颈诊断链
    engine.build_chain(
        "comm_bottleneck_diagnosis",
        "通信瓶颈诊断链：分析集合操作 → 检查重叠率 → 检测抖动"
    ).add_step(
        "analyze_collective_ops",
        inputs={},
    ).add_step(
        "check_overlap_ratio",
        inputs={},
    ).add_step(
        "detect_comm_jitter",
        inputs={},
    )
    
    # 慢卡诊断链
    engine.build_chain(
        "slow_rank_diagnosis",
        "慢卡诊断链：检测慢卡 → 分析跨卡抖动 → 定位原因"
    ).add_step(
        "detect_slow_rank",
        inputs={},
    ).add_step(
        "analyze_cross_rank_jitter",
        inputs={},
    )


def get_engine() -> SkillEngine:
    """获取全局技能引擎实例"""
    engine = SkillEngine()
    create_builtin_chains(engine)
    return engine
