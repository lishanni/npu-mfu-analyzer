"""
NPU MFU Analyzer MCP Server

通过 MCP (Model Context Protocol) 将 Skills 暴露给 Claude Code。
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP SDK
from mcp.server import Server
from mcp.types import Tool, TextContent

from .sdk import NPUMFUAnalyzerSDK
from .base import SkillContext

logger = logging.getLogger(__name__)

# 创建 MCP Server
app = Server("npu-mfu-analyzer")

# 初始化 SDK
sdk: Optional[NPUMFUAnalyzerSDK] = None


def get_sdk() -> NPUMFUAnalyzerSDK:
    """获取或创建 SDK 实例"""
    global sdk
    if sdk is None:
        sdk = NPUMFUAnalyzerSDK()
    return sdk


@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出所有可用的 Skills 作为 MCP Tools"""
    sdk = get_sdk()
    skills = sdk.list_skills()

    tools = []
    for skill_meta in skills:
        # 构建 JSON Schema 输入定义
        properties = {}
        required = []

        for inp in skill_meta.get("inputs", []):
            prop = {"type": inp["type"], "description": inp["description"]}
            if inp.get("default"):
                prop["default"] = inp["default"]
            properties[inp["name"]] = prop
            if inp.get("required", True):
                required.append(inp["name"])

        # 添加 profiling_path 参数
        properties["profiling_path"] = {
            "type": "string",
            "description": "Profiling 数据目录路径"
        }
        required.append("profiling_path")

        tool = Tool(
            name=skill_meta["name"],
            description=skill_meta["description"],
            inputSchema={
                "type": "object",
                "properties": properties,
                "required": required,
            }
        )
        tools.append(tool)

    # 添加复合工具
    tools.extend([
        Tool(
            name="analyze_profiling",
            description="执行完整的 Profiling 分析，包括 MFU、时间线、通信等",
            inputSchema={
                "type": "object",
                "properties": {
                    "profiling_path": {
                        "type": "string",
                        "description": "Profiling 数据目录路径"
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要执行的技能列表（可选）"
                    }
                },
                "required": ["profiling_path"]
            }
        ),
        Tool(
            name="compare_profiling",
            description="对比分析两个 Profiling 数据，找出性能差异根因",
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline_path": {
                        "type": "string",
                        "description": "基准 Profiling 数据路径"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "目标 Profiling 数据路径"
                    }
                },
                "required": ["baseline_path", "target_path"]
            }
        ),
        Tool(
            name="list_available_skills",
            description="列出所有可用的分析技能",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ])

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """执行 Skill"""
    try:
        sdk = get_sdk()

        if name == "list_available_skills":
            skills = sdk.list_skills()
            result_text = "## 可用技能列表\n\n"
            for skill in skills:
                result_text += f"### {skill['display_name']}\n"
                result_text += f"- **名称**: `{skill['name']}`\n"
                result_text += f"- **类型**: {skill['skill_type']}\n"
                result_text += f"- **分类**: {skill['category']}\n"
                result_text += f"- **描述**: {skill['description']}\n\n"
            return [TextContent(type="text", text=result_text)]

        elif name == "analyze_profiling":
            profiling_path = arguments["profiling_path"]
            skills = arguments.get("skills")

            if not Path(profiling_path).exists():
                return [TextContent(type="text", text=f"错误: 路径不存在 {profiling_path}")]

            result = sdk.analyze(profiling_path)

            if result.get("success"):
                output = result.get("markdown", json.dumps(result, indent=2, ensure_ascii=False))
            else:
                output = f"分析失败: {result}"

            return [TextContent(type="text", text=output)]

        elif name == "compare_profiling":
            baseline_path = arguments["baseline_path"]
            target_path = arguments["target_path"]

            if not Path(baseline_path).exists():
                return [TextContent(type="text", text=f"错误: 基准路径不存在 {baseline_path}")]
            if not Path(target_path).exists():
                return [TextContent(type="text", text=f"错误: 目标路径不存在 {target_path}")]

            result = sdk.compare(baseline_path, target_path)

            if result.get("success"):
                output = result.get("markdown", json.dumps(result, indent=2, ensure_ascii=False))
            else:
                output = f"对比分析失败: {result}"

            return [TextContent(type="text", text=output)]

        else:
            # 执行单个 Skill
            profiling_path = arguments.get("profiling_path")
            if not profiling_path:
                return [TextContent(type="text", text="错误: 缺少 profiling_path 参数")]

            if not Path(profiling_path).exists():
                return [TextContent(type="text", text=f"错误: 路径不存在 {profiling_path}")]

            # 构建上下文
            kwargs = {k: v for k, v in arguments.items() if k != "profiling_path"}
            result = sdk.execute_skill(name, profiling_path, **kwargs)

            # 格式化输出
            if result.get("success"):
                output_lines = [
                    f"## {result.get('skill_name', name)}",
                    "",
                    f"**摘要**: {result.get('summary', 'N/A')}",
                    "",
                ]

                if result.get("data"):
                    output_lines.append("### 数据")
                    for k, v in result["data"].items():
                        output_lines.append(f"- **{k}**: {v}")
                    output_lines.append("")

                if result.get("recommendations"):
                    output_lines.append("### 建议")
                    for r in result["recommendations"]:
                        output_lines.append(f"- {r}")
                    output_lines.append("")

                if result.get("root_cause"):
                    output_lines.append(f"**根因**: {result['root_cause']}")

                output = "\n".join(output_lines)
            else:
                output = f"执行失败: {result.get('error', '未知错误')}"

            return [TextContent(type="text", text=output)]

    except Exception as e:
        logger.exception(f"执行 Skill 失败: {e}")
        return [TextContent(type="text", text=f"执行错误: {str(e)}")]


def run_server():
    """运行 MCP Server"""
    import asyncio
    from mcp.server.stdio import stdio_server

    asyncio.run(stdio_server(app))


if __name__ == "__main__":
    run_server()