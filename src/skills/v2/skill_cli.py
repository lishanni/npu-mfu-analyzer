"""
NPU MFU Analyzer CLI for Claude Code Integration

提供命令行接口，便于 Claude Code 调用。
"""

import json
import sys
import click
from pathlib import Path
from typing import Optional


@click.group()
def cli():
    """NPU MFU Analyzer - 昇腾 NPU 性能分析工具"""
    pass


@cli.command("list-skills")
def list_skills():
    """列出所有可用技能"""
    from src.skills.v2.sdk import NPUMFUAnalyzerSDK

    sdk = NPUMFUAnalyzerSDK()
    skills = sdk.list_skills()

    click.echo("## 可用技能列表\n")
    for skill in skills:
        click.echo(f"### {skill['display_name']}")
        click.echo(f"- **名称**: `{skill['name']}`")
        click.echo(f"- **类型**: {skill['skill_type']}")
        click.echo(f"- **分类**: {skill['category']}")
        click.echo(f"- **描述**: {skill['description']}")
        click.echo("")


@cli.command("analyze")
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option("--skill", "-s", multiple=True, help="指定执行的技能")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), default="markdown", help="输出格式")
def analyze(profiling_path: str, skill: tuple, output: Optional[str], format: str):
    """分析 Profiling 数据"""
    from src.skills.v2.sdk import NPUMFUAnalyzerSDK, AnalysisOptions

    sdk = NPUMFUAnalyzerSDK()

    skills = list(skill) if skill else None
    options = AnalysisOptions(
        skills=skills,
        output_format=format,
    )

    result = sdk.analyze(profiling_path, options)

    if format == "json":
        output_text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output_text = result.get("markdown", str(result))

    if output:
        Path(output).write_text(output_text)
        click.echo(f"结果已保存到: {output}")
    else:
        click.echo(output_text)


@cli.command("compare")
@click.argument("baseline_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), default="markdown", help="输出格式")
def compare(baseline_path: str, target_path: str, output: Optional[str], format: str):
    """对比分析两个 Profiling 数据"""
    from src.skills.v2.sdk import NPUMFUAnalyzerSDK, AnalysisOptions

    sdk = NPUMFUAnalyzerSDK()
    options = AnalysisOptions(output_format=format)

    result = sdk.compare(baseline_path, target_path, options)

    if format == "json":
        output_text = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output_text = result.get("markdown", str(result))

    if output:
        Path(output).write_text(output_text)
        click.echo(f"结果已保存到: {output}")
    else:
        click.echo(output_text)


@cli.command("execute")
@click.argument("skill_name")
@click.argument("profiling_path", type=click.Path(exists=True))
@click.option("--param", "-p", multiple=True, help="额外参数 (key=value)")
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), default="markdown", help="输出格式")
def execute(skill_name: str, profiling_path: str, param: tuple, format: str):
    """执行单个技能"""
    from src.skills.v2.sdk import NPUMFUAnalyzerSDK

    sdk = NPUMFUAnalyzerSDK()

    # 解析参数
    kwargs = {}
    for p in param:
        if "=" in p:
            key, value = p.split("=", 1)
            kwargs[key] = value

    result = sdk.execute_skill(skill_name, profiling_path, **kwargs)

    if format == "json":
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Markdown 格式
        lines = [
            f"## {result.get('skill_name', skill_name)}",
            "",
            f"**状态**: {'✅ 成功' if result.get('success') else '❌ 失败'}",
        ]

        if result.get("summary"):
            lines.append(f"**摘要**: {result['summary']}")

        if result.get("data"):
            lines.extend(["", "### 数据"])
            for k, v in result["data"].items():
                lines.append(f"- **{k}**: {v}")

        if result.get("recommendations"):
            lines.extend(["", "### 建议"])
            for r in result["recommendations"]:
                lines.append(f"- {r}")

        if result.get("root_cause"):
            lines.extend(["", f"**根因**: {result['root_cause']}"])

        if result.get("error"):
            lines.extend(["", f"**错误**: {result['error']}"])

        click.echo("\n".join(lines))


@cli.command("skill-info")
@click.argument("skill_name")
def skill_info(skill_name: str):
    """获取技能详细信息"""
    from src.skills.v2.sdk import NPUMFUAnalyzerSDK

    sdk = NPUMFUAnalyzerSDK()
    info = sdk.get_skill(skill_name)

    if not info:
        click.echo(f"技能 '{skill_name}' 不存在", err=True)
        sys.exit(1)

    click.echo(f"## {info['display_name']}\n")
    click.echo(f"- **名称**: `{info['name']}`")
    click.echo(f"- **类型**: {info['skill_type']}")
    click.echo(f"- **分类**: {info['category']}")
    click.echo(f"- **优先级**: {info['priority']}")
    click.echo(f"- **版本**: {info['version']}")
    click.echo(f"- **标签**: {', '.join(info['tags'])}")
    click.echo(f"\n**描述**: {info['description']}")

    if info.get("inputs"):
        click.echo("\n### 输入参数")
        for inp in info["inputs"]:
            required = " (必需)" if inp.get("required") else ""
            click.echo(f"- `{inp['name']}` ({inp['type']}){required}: {inp['description']}")

    if info.get("outputs"):
        click.echo("\n### 输出")
        for out in info["outputs"]:
            click.echo(f"- `{out['name']}` ({out['type']}): {out['description']}")

    if info.get("dependencies"):
        click.echo(f"\n**依赖技能**: {', '.join(info['dependencies'])}")


if __name__ == "__main__":
    cli()