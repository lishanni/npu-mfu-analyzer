"""
报告模板

定义 Markdown 和 HTML 报告模板。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReportData:
    """报告数据"""
    title: str = "昇腾 NPU 性能分析报告"
    generated_at: str = ""
    profiling_path: str = ""
    
    # 概览
    rank_count: int = 1
    step_count: int = 0
    avg_step_time_ms: float = 0.0
    estimated_mfu: float = 0.0
    
    # 时间分布
    compute_ratio: float = 0.0
    comm_ratio: float = 0.0
    idle_ratio: float = 0.0
    overlap_ratio: float = 0.0
    
    # 瓶颈
    main_bottleneck: str = ""
    bottleneck_impact: float = 0.0
    
    # 详细分析
    timeline_analysis: str = ""
    operator_analysis: str = ""
    memory_analysis: str = ""
    communication_analysis: str = ""
    
    # 建议
    suggestions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.suggestions is None:
            self.suggestions = []


class MarkdownTemplate:
    """Markdown 报告模板"""
    
    TEMPLATE = """# {title}

**生成时间**: {generated_at}
**数据路径**: `{profiling_path}`

---

## 1. 性能概览

| 指标 | 数值 |
|------|------|
| Rank 数量 | {rank_count} |
| Step 数量 | {step_count} |
| 平均 Step 时间 | {avg_step_time_ms:.2f} ms |
| 估算 MFU | {mfu_pct:.1f}% |

### 时间分布

```
计算: {compute_bar} {compute_pct:.1f}%
通信: {comm_bar} {comm_pct:.1f}%
空闲: {idle_bar} {idle_pct:.1f}%
```

### 主要瓶颈

**{main_bottleneck}** (影响 {bottleneck_impact:.1f}%)

---

## 2. 详细分析

### 2.1 Timeline 分析

{timeline_analysis}

### 2.2 算子分析

{operator_analysis}

### 2.3 内存分析

{memory_analysis}

### 2.4 通信分析

{communication_analysis}

---

## 3. 优化建议

{suggestions_section}

---

## 4. 附录

### 4.1 数据来源

- Profiling 工具: msprof
- 分析工具: npu-mfu-analyzer

### 4.2 术语解释

| 术语 | 说明 |
|------|------|
| MFU | Model FLOPS Utilization，模型算力利用率 |
| TP | Tensor Parallel，张量并行 |
| PP | Pipeline Parallel，流水线并行 |
| DP | Data Parallel，数据并行 |
| Overlap | 通信掩盖，通信与计算重叠的比例 |
| Bubble | PP 中的气泡时间，Stage 等待的空闲时间 |
"""
    
    @classmethod
    def render(cls, data: ReportData) -> str:
        """渲染 Markdown 报告"""
        
        # 生成进度条
        def make_bar(pct: float, width: int = 20) -> str:
            filled = int(pct / 100 * width)
            return "█" * filled + "░" * (width - filled)
        
        compute_bar = make_bar(data.compute_ratio * 100)
        comm_bar = make_bar(data.comm_ratio * 100)
        idle_bar = make_bar(data.idle_ratio * 100)
        
        # 生成建议部分
        suggestions_section = cls._render_suggestions(data.suggestions)
        
        return cls.TEMPLATE.format(
            title=data.title,
            generated_at=data.generated_at,
            profiling_path=data.profiling_path,
            rank_count=data.rank_count,
            step_count=data.step_count,
            avg_step_time_ms=data.avg_step_time_ms,
            mfu_pct=data.estimated_mfu * 100,
            compute_bar=compute_bar,
            compute_pct=data.compute_ratio * 100,
            comm_bar=comm_bar,
            comm_pct=data.comm_ratio * 100,
            idle_bar=idle_bar,
            idle_pct=data.idle_ratio * 100,
            main_bottleneck=data.main_bottleneck or "无明显瓶颈",
            bottleneck_impact=data.bottleneck_impact,
            timeline_analysis=data.timeline_analysis or "暂无分析数据",
            operator_analysis=data.operator_analysis or "暂无分析数据",
            memory_analysis=data.memory_analysis or "暂无分析数据",
            communication_analysis=data.communication_analysis or "暂无分析数据",
            suggestions_section=suggestions_section,
        )
    
    @classmethod
    def _render_suggestions(cls, suggestions: List[Dict[str, Any]]) -> str:
        """渲染建议部分"""
        if not suggestions:
            return "暂无优化建议。"
        
        lines = []
        priority_groups = {"high": [], "medium": [], "low": []}
        
        for s in suggestions:
            priority = s.get("priority", "medium")
            priority_groups.get(priority, priority_groups["medium"]).append(s)
        
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        priority_names = {"high": "高优先级", "medium": "中优先级", "low": "低优先级"}
        
        for priority in ["high", "medium", "low"]:
            group = priority_groups[priority]
            if group:
                emoji = priority_emoji[priority]
                name = priority_names[priority]
                lines.append(f"### {emoji} {name}")
                lines.append("")
                
                for i, s in enumerate(group, 1):
                    title = s.get("title", "未命名建议")
                    desc = s.get("description", "")
                    benefit = s.get("expected_benefit", "")
                    
                    lines.append(f"**{i}. {title}**")
                    if desc:
                        lines.append(f"")
                        lines.append(desc)
                    if benefit:
                        lines.append(f"")
                        lines.append(f"*预期收益*: {benefit}")
                    
                    code = s.get("code_example", "")
                    if code:
                        lines.append("")
                        lines.append("```python")
                        lines.append(code)
                        lines.append("```")
                    
                    lines.append("")
        
        return "\n".join(lines)


class HTMLTemplate:
    """HTML 报告模板"""
    
    TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #1a73e8;
            --success-color: #34a853;
            --warning-color: #fbbc04;
            --danger-color: #ea4335;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #202124;
            --border-color: #dadce0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-color), #4285f4);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        
        .header .meta {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary-color);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: var(--bg-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 4px;
        }}
        
        .progress-bar {{
            height: 24px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            margin: 8px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            color: white;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .progress-compute {{ background: var(--success-color); }}
        .progress-comm {{ background: var(--warning-color); }}
        .progress-idle {{ background: var(--danger-color); }}
        
        .suggestion {{
            border-left: 4px solid var(--border-color);
            padding: 16px;
            margin: 12px 0;
            background: var(--bg-color);
            border-radius: 0 8px 8px 0;
        }}
        
        .suggestion.high {{ border-left-color: var(--danger-color); }}
        .suggestion.medium {{ border-left-color: var(--warning-color); }}
        .suggestion.low {{ border-left-color: var(--success-color); }}
        
        .suggestion h4 {{
            margin-bottom: 8px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .badge.high {{ background: var(--danger-color); color: white; }}
        .badge.medium {{ background: var(--warning-color); color: black; }}
        .badge.low {{ background: var(--success-color); color: white; }}
        
        pre {{
            background: #263238;
            color: #aabbc3;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 13px;
            margin-top: 12px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-color);
            font-weight: 600;
        }}
        
        .bottleneck-alert {{
            background: #fce8e6;
            border: 1px solid var(--danger-color);
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }}
        
        .bottleneck-alert h4 {{
            color: var(--danger-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 {title}</h1>
            <div class="meta">
                生成时间: {generated_at} | 
                数据路径: {profiling_path}
            </div>
        </div>
        
        <div class="card">
            <h2>📊 性能概览</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{rank_count}</div>
                    <div class="metric-label">Rank 数量</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{step_count}</div>
                    <div class="metric-label">Step 数量</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_step_time_ms:.1f}ms</div>
                    <div class="metric-label">平均 Step 时间</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{mfu_pct:.1f}%</div>
                    <div class="metric-label">估算 MFU</div>
                </div>
            </div>
            
            <h3>时间分布</h3>
            <div>
                <label>计算 ({compute_pct:.1f}%)</label>
                <div class="progress-bar">
                    <div class="progress-fill progress-compute" style="width: {compute_pct}%">{compute_pct:.0f}%</div>
                </div>
            </div>
            <div>
                <label>通信 ({comm_pct:.1f}%)</label>
                <div class="progress-bar">
                    <div class="progress-fill progress-comm" style="width: {comm_pct}%">{comm_pct:.0f}%</div>
                </div>
            </div>
            <div>
                <label>空闲 ({idle_pct:.1f}%)</label>
                <div class="progress-bar">
                    <div class="progress-fill progress-idle" style="width: {idle_pct}%">{idle_pct:.0f}%</div>
                </div>
            </div>
            
            {bottleneck_section}
        </div>
        
        <div class="card">
            <h2>💡 优化建议</h2>
            {suggestions_section}
        </div>
        
        <div class="card">
            <h2>📈 详细分析</h2>
            {analysis_section}
        </div>
    </div>
</body>
</html>
"""
    
    @classmethod
    def render(cls, data: ReportData) -> str:
        """渲染 HTML 报告"""
        
        # 瓶颈部分
        bottleneck_section = ""
        if data.main_bottleneck:
            bottleneck_section = f"""
            <div class="bottleneck-alert">
                <h4>⚠️ 主要瓶颈: {data.main_bottleneck}</h4>
                <p>影响程度: {data.bottleneck_impact:.1f}%</p>
            </div>
            """
        
        # 建议部分
        suggestions_section = cls._render_suggestions(data.suggestions)
        
        # 分析部分
        analysis_section = f"""
        <h3>Timeline 分析</h3>
        <p>{data.timeline_analysis or '暂无数据'}</p>
        <h3>算子分析</h3>
        <p>{data.operator_analysis or '暂无数据'}</p>
        """
        
        return cls.TEMPLATE.format(
            title=data.title,
            generated_at=data.generated_at,
            profiling_path=data.profiling_path,
            rank_count=data.rank_count,
            step_count=data.step_count,
            avg_step_time_ms=data.avg_step_time_ms,
            mfu_pct=data.estimated_mfu * 100,
            compute_pct=data.compute_ratio * 100,
            comm_pct=data.comm_ratio * 100,
            idle_pct=data.idle_ratio * 100,
            bottleneck_section=bottleneck_section,
            suggestions_section=suggestions_section,
            analysis_section=analysis_section,
        )
    
    @classmethod
    def _render_suggestions(cls, suggestions: List[Dict[str, Any]]) -> str:
        """渲染建议 HTML"""
        if not suggestions:
            return "<p>暂无优化建议。</p>"
        
        html_parts = []
        for s in suggestions:
            priority = s.get("priority", "medium")
            title = s.get("title", "")
            desc = s.get("description", "")
            benefit = s.get("expected_benefit", "")
            code = s.get("code_example", "")
            
            html_parts.append(f"""
            <div class="suggestion {priority}">
                <h4><span class="badge {priority}">{priority.upper()}</span> {title}</h4>
                {f'<p>{desc}</p>' if desc else ''}
                {f'<p><strong>预期收益:</strong> {benefit}</p>' if benefit else ''}
                {f'<pre>{code}</pre>' if code else ''}
            </div>
            """)
        
        return "\n".join(html_parts)
