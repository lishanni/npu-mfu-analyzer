"""
AIKG 融合算子生成工作流

独立的融合算子生成工作流模块，负责：
1. 分析 Profiling 数据
2. 检测融合机会
3. 调用 AIKG 生成 Triton-Ascend 代码
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader
from npu_mfu_analyzer.llm.llm_interface import LLMConfig, LLMFactory
from npu_mfu_analyzer.agents.operator_agent import OperatorAgent


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """生成结果"""
    success: bool
    fusion_opportunities: List[Dict[str, Any]]
    generated_kernels: List[Dict[str, Any]]
    output_dir: Path
    error: Optional[str] = None


class AIKGWorkflow:
    """
    AIKG 融合算子生成工作流

    完整的自动化工作流：从 Profiling 数据到融合算子代码
    """

    def __init__(
        self,
        profiling_path: str,
        output_dir: str = "./generated_kernels",
        backend: str = "claude",
        model: Optional[str] = None,
        min_speedup: float = 1.05,
        max_complexity: str = "高",
        skip_native: bool = True,
        timeout: int = 300,
        max_concurrent: int = 3,
    ):
        """
        初始化 AIKG 工作流

        Args:
            profiling_path: Profiling 数据目录路径
            output_dir: 融合算子输出目录
            backend: LLM 后端 (claude, openai, ollama, deepseek, mock)
            model: LLM 模型名称
            min_speedup: 最小加速比阈值
            max_complexity: 最大实现复杂度 (低/中等/高)
            skip_native: 是否跳过昇腾已有融合算子
            timeout: 单个算子生成超时时间（秒）
            max_concurrent: 最大并发生成数
        """
        self.profiling_path = profiling_path
        self.output_dir = Path(output_dir)
        self.backend = backend
        self.model = model
        self.min_speedup = min_speedup
        self.max_complexity = max_complexity
        self.skip_native = skip_native
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    async def run(self) -> GenerationResult:
        """
        执行完整的 AIKG 生成工作流

        Returns:
            GenerationResult: 包含融合机会、生成的内核等信息
        """
        try:
            logger.info("开始 AIKG 融合算子生成工作流")

            # 步骤 1: 加载 Profiling 数据
            logger.info("步骤 1/4: 加载 Profiling 数据")
            loader = ProfilingLoader(self.profiling_path)
            info = loader.detect()
            top_kernels = loader.get_top_kernels(top_n=30)
            logger.info(f"检测到 {info.framework} 框架，{len(top_kernels)} 个算子")

            # 步骤 2: 初始化 LLM 和 Agent
            logger.info("步骤 2/4: 初始化 LLM 和 Agent")

            # 构建 LLM 配置
            llm_config = LLMConfig(backend=self.backend)
            if self.model:
                llm_config.model = self.model

            # 设置 API Key（如果指定）
            if self.backend == "claude":
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    llm_config.api_key = api_key

            llm = LLMFactory.create(llm_config)

            # 构建 Agent 配置（启用 AIKG）
            agent_config = {
                "aikg_enabled": True,
                "aikg_output_dir": str(self.output_dir),
                "aikg_min_speedup": self.min_speedup,
                "aikg_max_complexity": self.max_complexity,
                "aikg_skip_native": self.skip_native,
                "aikg_timeout": self.timeout,
                "aikg_max_concurrent": self.max_concurrent,
            }

            agent = OperatorAgent(llm=llm, config=agent_config)
            logger.info(f"AIKG 已启用，输出目录: {self.output_dir}")

            # 步骤 3: 分析并生成融合算子
            logger.info("步骤 3/4: 分析融合机会并生成代码")

            # 准备分析数据
            analysis_data = {
                "top_operators": top_kernels,
                "profiling_path": self.profiling_path,
            }

            # 执行分析（包含 AIKG 生成）
            result = await agent.analyze(analysis_data)

            # 提取融合机会
            fusion_opportunities = []
            if result.details:
                fusion_opportunities = result.details.get("fusion_opportunities", [])

            logger.info(f"检测到 {len(fusion_opportunities)} 个融合机会")

            # 步骤 4: 收集生成的内核文件
            logger.info("步骤 4/4: 收集生成的文件")

            generated_kernels = self._collect_generated_kernels()
            logger.info(f"生成了 {len(generated_kernels)} 个融合算子")

            return GenerationResult(
                success=True,
                fusion_opportunities=fusion_opportunities,
                generated_kernels=generated_kernels,
                output_dir=self.output_dir,
            )

        except Exception as e:
            logger.error(f"AIKG 工作流失败: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                fusion_opportunities=[],
                generated_kernels=[],
                output_dir=self.output_dir,
                error=str(e),
            )

    def _collect_generated_kernels(self) -> List[Dict[str, Any]]:
        """收集生成的内核文件信息"""
        kernels = []

        if not self.output_dir.exists():
            return kernels

        # 查找生成的 Python 文件（排除 benchmark）
        for py_file in self.output_dir.glob("*.py"):
            if "bench" not in py_file.name:
                kernels.append({
                    "name": py_file.stem,
                    "file": str(py_file),
                    "lines": len(py_file.read_text(errors='ignore').split('\n')),
                })

        return kernels

    def print_summary(self, result: GenerationResult):
        """打印生成结果摘要（用于 CLI 输出）"""
        import click

        click.echo("\n" + "=" * 70)
        click.echo(click.style("📊 AIKG 生成结果摘要", fg="cyan", bold=True))
        click.echo("=" * 70)

        if not result.success:
            click.echo(click.style(f"❌ 生成失败: {result.error}", fg="red"))
            return

        fusion_opps = result.fusion_opportunities
        kernels = result.generated_kernels

        # 显示融合机会
        if fusion_opps:
            click.echo(click.style(f"\n✓ 检测到 {len(fusion_opps)} 个融合机会:", fg="green"))
            for i, opp in enumerate(fusion_opps[:5], 1):
                click.echo(f"  {i}. {opp['name']}")
                click.echo(f"     - 类型: {opp['type']} | 加速比: {opp['speedup']:.1f}x")
                click.echo(f"     - 端到端加速: {opp['end_to_end_speedup']:.1%} | 复杂度: {opp['complexity']}")
            if len(fusion_opps) > 5:
                click.echo(f"  ... 还有 {len(fusion_opps) - 5} 个融合机会")

        # 显示生成的内核
        if kernels:
            click.echo(click.style(f"\n✓ 成功生成 {len(kernels)} 个融合算子:", fg="green", bold=True))
            for kernel in kernels:
                click.echo(f"  📄 {kernel['name']}")
                click.echo(f"     - 文件: {kernel['file']}")
                click.echo(f"     - 代码行数: {kernel['lines']}")

            # 查找配套文件
            sh_files = list(result.output_dir.glob("*.sh"))
            bench_files = list(result.output_dir.glob("*_bench.py"))
            click.echo(f"\n  配套文件:")
            click.echo(f"  - 编译脚本: {len(sh_files)} 个")
            click.echo(f"  - 性能测试: {len(bench_files)} 个")

        click.echo("\n" + "=" * 70)
        click.echo(click.style("🎉 生成完成！", fg="green", bold=True))
        click.echo(click.style(f"📁 所有文件保存在: {result.output_dir}", fg="blue"))
        click.echo("=" * 70)

        # 显示使用提示
        if kernels:
            click.echo("\n💡 使用提示:")
            click.echo("   1. 查看生成的 Triton 代码:")
            click.echo(f"      cat {kernels[0]['file']}")
            click.echo("   2. 编译融合算子:")
            click.echo(f"      bash {result.output_dir}/*.sh")
            click.echo("   3. 运行性能测试:")
            click.echo(f"      python {result.output_dir}/*_bench.py")


async def run_aikg_workflow(**kwargs) -> GenerationResult:
    """
    便捷函数：运行 AIKG 工作流

    Args:
        **kwargs: 传递给 AIKGWorkflow 的参数

    Returns:
        GenerationResult
    """
    workflow = AIKGWorkflow(**kwargs)
    return await workflow.run()
