"""
CLI 入口 - 命令行接口

Usage:
    npu-analyzer analyze /path/to/profiling
    npu-analyzer analyze /path/to/profiling --output report.md
    npu-analyzer analyze /path/to/profiling --backend mock
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="npu-mfu-analyzer")
def cli():
    """NPU MFU Analyzer - 昇腾 NPU 大模型训练性能分析工具"""
    pass


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="输出报告路径（默认输出到终端）"
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["openai", "claude", "ollama", "deepseek", "mock"]),
    default="openai",
    help="LLM 后端: openai/claude/ollama/deepseek/mock"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="LLM 模型名称"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="详细输出"
)
def analyze(
    profiling_path: str,
    output: Optional[str],
    backend: str,
    model: Optional[str],
    verbose: bool
):
    """
    分析 Profiling 数据，生成性能报告
    
    PROFILING_PATH: Profiling 数据目录路径
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo(f"🚀 开始分析 Profiling 数据: {profiling_path}")
    click.echo(f"   LLM 后端: {backend}")
    
    # 运行分析
    try:
        report = asyncio.run(_run_analysis(profiling_path, backend, model))
        
        if report.success:
            click.echo(click.style("✅ 分析完成!", fg="green"))
            
            # 输出报告
            report_text = report.to_markdown()
            
            if output:
                output_path = Path(output)
                output_path.write_text(report_text, encoding="utf-8")
                click.echo(f"📄 报告已保存到: {output}")
            else:
                click.echo("\n" + "=" * 60)
                click.echo(report_text)
                click.echo("=" * 60)
            
            # 输出摘要
            if report.recommendations:
                click.echo(click.style("\n📌 Top 优化建议:", fg="yellow"))
                for i, rec in enumerate(report.recommendations[:5], 1):
                    click.echo(f"   {i}. {rec}")
        else:
            click.echo(click.style(f"❌ 分析失败: {report.error}", fg="red"))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"❌ 发生错误: {e}", fg="red"))
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def _run_analysis(profiling_path: str, backend: str, model: Optional[str]):
    """执行分析"""
    from src.llm.llm_interface import LLMConfig
    from src.agents.orchestrator import Orchestrator
    
    # 构建 LLM 配置
    config = LLMConfig(backend=backend)
    if model:
        config.model = model
    
    # 创建 Orchestrator 并运行
    orchestrator = Orchestrator(profiling_path, llm_config=config)
    return await orchestrator.run()


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
def info(profiling_path: str):
    """
    显示 Profiling 数据信息
    
    PROFILING_PATH: Profiling 数据目录路径
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    
    click.echo(f"📊 检测 Profiling 数据: {profiling_path}")
    
    loader = ProfilingLoader(profiling_path)
    info = loader.detect()
    
    click.echo(f"\n数据类型: {info.data_type}")
    click.echo(f"框架: {info.framework}")
    click.echo(f"Rank 数量: {info.rank_count}")
    click.echo(f"Timeline 数据: {'✓' if info.has_timeline else '✗'}")
    click.echo(f"Memory 数据: {'✓' if info.has_memory else '✗'}")
    click.echo(f"Communication 数据: {'✓' if info.has_communication else '✗'}")
    
    if info.db_paths:
        click.echo(f"\nDB 文件 ({len(info.db_paths)}):")
        for path in info.db_paths[:5]:
            click.echo(f"  - {path}")
        if len(info.db_paths) > 5:
            click.echo(f"  ... 还有 {len(info.db_paths) - 5} 个文件")
    
    if info.json_paths:
        click.echo(f"\nJSON 文件 ({len(info.json_paths)}):")
        for path in info.json_paths[:5]:
            click.echo(f"  - {path}")
        if len(info.json_paths) > 5:
            click.echo(f"  ... 还有 {len(info.json_paths) - 5} 个文件")


@cli.command()
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option("--max-steps", type=int, default=10, help="采样的 Step 数量")
def summary(profiling_path: str, max_steps: int):
    """
    生成 Profiling 数据摘要（不调用 LLM）
    
    PROFILING_PATH: Profiling 数据目录路径
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.data_loader.data_summarizer import DataSummarizer
    
    click.echo(f"📊 生成数据摘要: {profiling_path}")
    
    loader = ProfilingLoader(profiling_path)
    summarizer = DataSummarizer(loader)
    
    summary = summarizer.summarize(max_sample_steps=max_steps)
    
    click.echo("\n" + "=" * 60)
    click.echo(summary.to_prompt_text())
    click.echo("=" * 60)


@cli.command()
def version():
    """显示版本信息"""
    click.echo("NPU MFU Analyzer v0.1.0")
    click.echo("昇腾 NPU 大模型训练性能分析工具")


@cli.command()
@click.option(
    "--host", "-h",
    type=str,
    default="0.0.0.0",
    help="监听地址（默认 0.0.0.0）"
)
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="监听端口（默认 8000）"
)
@click.option(
    "--reload",
    is_flag=True,
    help="开发模式，自动重载"
)
def web(host: str, port: int, reload: bool):
    """
    启动 Web 服务
    
    启动后访问 http://localhost:8000 使用 Web 界面
    API 文档: http://localhost:8000/docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(click.style("❌ 请先安装 web 依赖: pip install fastapi uvicorn", fg="red"))
        sys.exit(1)
    
    click.echo(f"🚀 启动 Web 服务: http://{host}:{port}")
    click.echo(f"📚 API 文档: http://{host}:{port}/docs")
    click.echo("按 Ctrl+C 停止服务\n")
    
    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    cli()
