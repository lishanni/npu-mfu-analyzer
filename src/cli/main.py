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
    "--format", "-f",
    type=click.Choice(["markdown", "html", "md"]),
    default="markdown",
    help="报告格式: markdown/html (默认: markdown)"
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
    format: str,
    verbose: bool
):
    """
    分析 Profiling 数据，生成性能报告

    PROFILING_PATH: Profiling 数据目录路径

    示例:
        npu-analyzer analyze /path/to/profiling
        npu-analyzer analyze /path/to/profiling -b claude -m GLM-4.7
        npu-analyzer analyze /path/to/profiling -o report.md -f markdown
        npu-analyzer analyze /path/to/profiling -o report.html -f html
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 标准化 format 参数
    if format == "md":
        format = "markdown"

    click.echo(f"🚀 开始分析 Profiling 数据: {profiling_path}")
    click.echo(f"   LLM 后端: {backend}")
    click.echo(f"   报告格式: {format}")

    # 运行分析
    try:
        report = asyncio.run(_run_analysis(profiling_path, backend, model))

        if report.success:
            click.echo(click.style("✅ 分析完成!", fg="green"))

            # 根据格式输出报告
            if format == "html":
                output_path = _generate_html_report(profiling_path, output)
                if output_path:
                    click.echo(f"📄 HTML 报告已保存到: {output_path}")
                else:
                    click.echo(click.style("❌ HTML 报告生成失败", fg="red"))
                    sys.exit(1)
            else:
                # Markdown 格式
                report_text = report.to_markdown()

                if output:
                    output_path = Path(output)
                    # 确保文件扩展名正确
                    if not output_path.suffix or output_path.suffix not in ['.md', '.markdown']:
                        output_path = output_path.with_suffix('.md')
                    output_path.write_text(report_text, encoding="utf-8")
                    click.echo(f"📄 报告已保存到: {output_path}")
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


def _generate_html_report(profiling_path: str, output: Optional[str]) -> Optional[Path]:
    """
    生成 HTML 格式报告

    Args:
        profiling_path: Profiling 数据路径
        output: 输出路径（可选）

    Returns:
        输出文件路径，失败返回 None
    """
    from src.data_loader.profiling_loader import ProfilingLoader
    from src.agents.operator_agent import FusionAnalyzer
    from datetime import datetime

    try:
        # 1. 加载数据
        loader = ProfilingLoader(profiling_path)
        info = loader.detect()
        top_kernels = loader.get_top_kernels(top_n=100)

        # 2. 融合分析
        analyzer = FusionAnalyzer()
        fusion_opportunities = analyzer.detect_opportunities(
            all_operators=top_kernels,
            timeline_data=None
        )

        # 3. 生成 HTML
        html = _build_html_report(
            profiling_path,
            top_kernels,
            fusion_opportunities,
            info
        )

        # 4. 确定输出路径
        if output:
            output_path = Path(output)
            # 如果是目录，生成文件名
            if output_path.is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_path / f"npu_mfu_report_{timestamp}.html"
            # 确保有正确的扩展名
            elif output_path.suffix != '.html':
                output_path = output_path.with_suffix('.html')
        else:
            # 默认输出到当前目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"npu_mfu_report_{timestamp}.html")

        # 5. 保存文件
        output_path.write_text(html, encoding="utf-8")

        return output_path

    except Exception as e:
        logger.error(f"HTML 报告生成失败: {e}", exc_info=True)
        return None


def _build_html_report(profiling_path: str, top_kernels: list, fusion_opportunities: list, info) -> str:
    """构建 HTML 报告内容"""
    from datetime import datetime

    # 计算统计数据
    total_dur = sum(k.get("dur", 0) for k in top_kernels[:15])

    # 算子类型统计
    op_types = {}
    for kernel in top_kernels:
        name = kernel.get("name", "")
        if "matmul" in name.lower():
            op_types["MatMul"] = op_types.get("MatMul", 0) + kernel.get("dur", 0)
        elif "attention" in name.lower() or "attn" in name.lower():
            op_types["Attention"] = op_types.get("Attention", 0) + kernel.get("dur", 0)
        elif "flash" in name.lower():
            op_types["FlashAttention"] = op_types.get("FlashAttention", 0) + kernel.get("dur", 0)
        elif "norm" in name.lower():
            op_types["Norm"] = op_types.get("Norm", 0) + kernel.get("dur", 0)
        elif "add" in name.lower():
            op_types["Add"] = op_types.get("Add", 0) + kernel.get("dur", 0)
        elif "gelu" in name.lower() or "relu" in name.lower() or "silu" in name.lower():
            op_types["Activation"] = op_types.get("Activation", 0) + kernel.get("dur", 0)
        elif "cast" in name.lower():
            op_types["Cast"] = op_types.get("Cast", 0) + kernel.get("dur", 0)
        else:
            op_types["Other"] = op_types.get("Other", 0) + kernel.get("dur", 0)

    total_type_dur = sum(dur for dur in op_types.values())

    # 生成 Top 15 表格行
    table_rows = []
    for i, kernel in enumerate(top_kernels[:15], 1):
        name = kernel.get("name", "unknown")[:60]
        dur_ms = kernel.get("dur", 0) / 1000
        ratio = (kernel.get("dur", 0) / total_dur * 100) if total_dur > 0 else 0
        table_rows.append(f'                <tr><td>{i}</td><td><code>{name}</code></td><td>{dur_ms:.2f}</td><td>{ratio:.1f}%</td></tr>')
    table_rows_html = "\n".join(table_rows)

    # 生成算子类型分布
    type_rows = []
    sorted_types = sorted(op_types.items(), key=lambda x: x[1], reverse=True)
    for op_type, dur in sorted_types:
        ratio = (dur / total_type_dur * 100) if total_type_dur > 0 else 0
        type_rows.append(f'                <li><strong>{op_type}</strong>: {dur/1000:.2f} ms ({ratio:.1f}%)</li>')
    type_rows_html = "\n".join(type_rows)

    # 生成融合机会卡片
    fusion_cards = []
    for i, opp in enumerate(fusion_opportunities, 1):
        priority_class = "high-priority" if opp.end_to_end_speedup >= 1.05 else "medium-priority"
        speedup_percent = (opp.end_to_end_speedup - 1) * 100

        ops_list = "\n".join([
            f'                    <li><code>{op.get("name", "unknown")}</code>: {op.get("dur", 0)/1000:.2f} ms</li>'
            for op in opp.current_ops[:5]
        ])
        if len(opp.current_ops) > 5:
            ops_list += f'\n                    <li><em>... 还有 {len(opp.current_ops) - 5} 个算子</em></li>'

        ascend_op_line = f"                <p><strong>昇腾算子</strong>: <code>{opp.ascend_op}</code></p>\n" if opp.ascend_op else ""

        fusion_cards.append(f'''
            <div class="card {priority_class}">
                <h3>{i}. {opp.name}</h3>
                <p><strong>类型</strong>: {opp.opportunity_type} | <strong>复杂度</strong>: {opp.complexity}</p>
                <p><strong>算子加速</strong>: {opp.estimated_speedup:.1f}x | <strong>端到端加速</strong>: {speedup_percent:.1f}%</p>
                <p><strong>时间占比</strong>: {opp.time_proportion:.1%} | <strong>内存节省</strong>: {opp.memory_saving*100:.0f}%</p>
{ascend_op_line}                <p><strong>实现方式</strong>: {opp.implementation}</p>
                <p><strong>涉及算子</strong> ({len(opp.current_ops)} 个):</p>
                <ul>
{ops_list}
                </ul>
            </div>''')

    fusion_cards_html = "\n".join(fusion_cards)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NPU MFU 性能分析报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}

        .header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .section h3 {{
            font-size: 1.4em;
            color: #764ba2;
            margin-bottom: 15px;
            margin-top: 25px;
        }}

        .card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }}

        .card.high-priority {{
            border-left-color: #f59e0b;
            background: #fff8f0;
        }}

        .card.medium-priority {{
            border-left-color: #10b981;
            background: #f0fdf4;
        }}

        .card h3 {{
            margin-top: 0;
            color: #333;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .stat-box .value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-box .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}

        li {{
            margin: 5px 0;
        }}

        .footer {{
            background: #f9fafb;
            padding: 20px;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NPU MFU 性能分析报告</h1>
            <div class="meta">
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据路径: <code>{profiling_path}</code></p>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>1. 概述</h2>
                <p>本报告基于真实 NPU Profiling 数据，分析算子性能、MFU（Model FLOPS Utilization）以及<strong>算子融合优化机会</strong>。</p>

                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="value">{info.data_type}</div>
                        <div class="label">数据类型</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{info.framework or 'Unknown'}</div>
                        <div class="label">框架</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{len(top_kernels)}</div>
                        <div class="label">算子数量</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">{len(fusion_opportunities)}</div>
                        <div class="label">融合机会</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>2. 算子性能分析</h2>
                <h3>Top 15 耗时算子</h3>
                <table>
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>算子名称</th>
                            <th>耗时 (ms)</th>
                            <th>占比</th>
                        </tr>
                    </thead>
                    <tbody>
{table_rows_html}
                    </tbody>
                </table>

                <h3>算子类型分布</h3>
                <ul>
{type_rows_html}
                </ul>
            </div>

            <div class="section">
                <h2>3. 算子融合机会分析</h2>
                <p>检测到 <strong>{len(fusion_opportunities)}</strong> 个融合机会，按端到端加速效果排序：</p>
{fusion_cards_html}
            </div>

            <div class="section">
                <h2>4. 优化建议</h2>
                <h3>4.1 高优先级优化 (端到端加速 > 5%)</h3>
'''

    # 添加高优先级优化建议
    high_priority = [opp for opp in fusion_opportunities if opp.end_to_end_speedup >= 1.05]
    if high_priority:
        for i, opp in enumerate(high_priority, 1):
            speedup_percent = (opp.end_to_end_speedup - 1) * 100
            html += f'''
                <div class="card high-priority">
                    <h3>{i}. {opp.name}</h3>
                    <p><strong>预期收益</strong>: 端到端加速 {speedup_percent:.1f}% (时间占比 {opp.time_proportion:.1%})</p>
                    <p><strong>实现难度</strong>: {opp.complexity}</p>
                    <p><strong>操作方式</strong>: {opp.implementation}</p>
                </div>'''
    else:
        html += '<p>暂无高优先级优化建议。</p>'

    # 添加通用优化建议
    html += '''
                <h3>4.2 优化方向建议</h3>
                <ul>
                    <li><strong>算子融合</strong>: 使用昇腾原生融合算子（如 FlashAttention、FusedMatMulBiasAct）</li>
                    <li><strong>数据类型优化</strong>: 统一使用 FP16/BF16 数据类型，减少不必要的数据类型转换</li>
                    <li><strong>内存布局优化</strong>: 减少 reshape/transpose 操作，优化张量内存布局</li>
                    <li><strong>数据加载优化</strong>: 增加 DataLoader workers，使用 pin_memory</li>
                </ul>
            </div>

            <div class="section">
                <h2>5. 技术细节</h2>
                <h3>5.1 融合分析方法</h3>
                <ul>
                    <li><strong>全局分析</strong>: 分析所有算子，不局限于 Top 10</li>
                    <li><strong>端到端加速计算</strong>: T_new = T_total - t_op + t_op/speedup</li>
                    <li><strong>融合模式匹配</strong>: 基于算子名称和执行序列</li>
                </ul>

                <h3>5.2 昇腾融合算子映射</h3>
                <table>
                    <thead>
                        <tr>
                            <th>融合模式</th>
                            <th>昇腾算子</th>
                            <th>性能提升</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>FlashAttention</td><td><code>aclnnFlashAttentionScore</code></td><td>5x+</td></tr>
                        <tr><td>MatMul+Bias+Act</td><td><code>aclnnFusedMatMulBiasAct</code></td><td>20-30%</td></tr>
                        <tr><td>Norm+Residual</td><td><code>aclnnAddRmsNorm</code></td><td>20%+</td></tr>
                        <tr><td>QKV Projection</td><td><code>aclnnFusedQKVProjection</code></td><td>2x</td></tr>
                        <tr><td>MoE 融合</td><td><code>aclnnGroupedMatmulV4</code></td><td>MoE 优化</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            <p>本报告由 npu-mfu-analyzer 自动生成</p>
            <p>Profiling 工具: msprof | 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>'''

    return html


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
