"""
通信矩阵可视化模块

生成通信矩阵热力图的 HTML 可视化。
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrix,
    LinkMetrics,
    TransportType,
)

logger = logging.getLogger(__name__)


class CommunicationMatrixVisualizer:
    """
    通信矩阵可视化器

    生成 HTML 格式的通信矩阵热力图，包括：
    - 带宽利用率热力图
    - 通信量热力图
    - 慢链路高亮
    - 传输类型标注
    """

    # 颜色配置
    COLORS = {
        "hccs": "#4CAF50",      # 绿色 - 节点内
        "rdma": "#2196F3",       # 蓝色 - 跨节点
        "roce": "#9C27B0",       # 紫色 - RoCE
        "pcie": "#FF9800",       # 橙色 - PCIe
        "slow_link": "#F44336",  # 红色 - 慢链路
        "bottleneck": "#E91E63", # 粉色 - 瓶颈链路
    }

    def __init__(self, matrix: CommunicationMatrix):
        """
        Args:
            matrix: CommunicationMatrix 对象
        """
        self.matrix = matrix

    def generate_html(
        self,
        title: str = "通信矩阵分析",
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成完整的 HTML 可视化页面

        Args:
            title: 页面标题
            output_path: 输出路径（可选）

        Returns:
            HTML 内容字符串
        """
        html = self._generate_html_content(title)

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"Communication matrix visualization saved to: {output_path}")

        return html

    def _generate_html_content(self, title: str) -> str:
        """生成 HTML 内容"""
        world_size = self.matrix.world_size
        matrix_2d = self.matrix.get_matrix_2d()
        util_2d = self.matrix.get_utilization_matrix() if hasattr(self.matrix, 'get_utilization_matrix') else None

        # 生成热力图数据
        bandwidth_data = self._generate_heatmap_data(matrix_2d, "bandwidth")
        utilization_data = self._generate_heatmap_data(
            util_2d if util_2d else matrix_2d, "utilization"
        )

        # 生成链路详情表格
        link_table = self._generate_link_table()

        # 生成慢链路列表
        slow_links_html = self._generate_slow_links_section()

        # 生成统计摘要
        summary_html = self._generate_summary_section()

        return f"""<!DOCTYPE html>
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
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
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
            font-size: 28px;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 4px;
        }}

        .heatmap-container {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .heatmap-wrapper {{
            flex: 1;
            min-width: 400px;
            max-width: 600px;
        }}

        .heatmap-wrapper h3 {{
            text-align: center;
            margin-bottom: 15px;
            color: #333;
        }}

        .heatmap {{
            display: grid;
            gap: 2px;
            margin: 0 auto;
        }}

        .heatmap-cell {{
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 500;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .heatmap-cell:hover {{
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10;
        }}

        .heatmap-cell.diagonal {{
            background: #e0e0e0 !important;
            color: #999;
        }}

        .heatmap-label {{
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: #666;
        }}

        .axis-label {{
            text-align: center;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            margin: 10px 0;
        }}

        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}

        .color-scale {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 15px;
            justify-content: center;
        }}

        .color-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #e8f5e9, #4CAF50, #FFC107, #F44336);
            border-radius: 4px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--bg-color);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}

        .table-container {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}

        .badge.hccs {{ background: #E8F5E9; color: #2E7D32; }}
        .badge.rdma {{ background: #E3F2FD; color: #1565C0; }}
        .badge.roce {{ background: #F3E5F5; color: #7B1FA2; }}
        .badge.pcie {{ background: #FFF3E0; color: #E65100; }}
        .badge.slow {{ background: #FFEBEE; color: #C62828; }}
        .badge.bottleneck {{ background: #FCE4EC; color: #AD1457; }}

        .alert {{
            padding: 16px;
            border-radius: 8px;
            margin: 16px 0;
        }}

        .alert-warning {{
            background: #FFF8E1;
            border-left: 4px solid #FFC107;
        }}

        .alert-danger {{
            background: #FFEBEE;
            border-left: 4px solid #F44336;
        }}

        .alert h4 {{
            margin-bottom: 8px;
        }}

        .transport-indicator {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 4px;
        }}

        .grid-{{world_size}} {{
            grid-template-columns: 50px repeat({world_size}, 50px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 {title}</h1>
            <div class="meta">
                World Size: {world_size} | 节点数: {self.matrix.num_machines or 1} | NPU/节点: {self.matrix.npus_per_machine or world_size}
            </div>
        </div>

        {summary_html}

        <div class="card">
            <h2>📊 通信矩阵热力图</h2>
            <div class="heatmap-container">
                <div class="heatmap-wrapper">
                    <h3>带宽利用率 (GB/s)</h3>
                    {bandwidth_data}
                    <div class="color-scale">
                        <span>低</span>
                        <div class="color-bar"></div>
                        <span>高</span>
                    </div>
                </div>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: {self.COLORS['hccs']}"></div>
                    <span>HCCS (节点内)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {self.COLORS['rdma']}"></div>
                    <span>RDMA (跨节点)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {self.COLORS['slow_link']}"></div>
                    <span>慢链路</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: {self.COLORS['bottleneck']}"></div>
                    <span>瓶颈链路</span>
                </div>
            </div>
        </div>

        {slow_links_html}

        <div class="card">
            <h2>📋 链路详情</h2>
            <div class="table-container">
                {link_table}
            </div>
        </div>
    </div>

    <script>
        // 鼠标悬停显示详情
        document.querySelectorAll('.heatmap-cell:not(.diagonal)').forEach(cell => {{
            cell.addEventListener('mouseenter', function() {{
                this.title = this.dataset.info || '';
            }});
        }});
    </script>
</body>
</html>"""

    def _generate_heatmap_data(self, matrix_2d: List[List[float]], data_type: str) -> str:
        """生成热力图 HTML"""
        world_size = self.matrix.world_size

        # 找到最大值用于颜色归一化
        max_val = max(max(row) for row in matrix_2d) if matrix_2d else 1
        if max_val == 0:
            max_val = 1

        rows = []

        # 表头行
        header_cells = ['<div class="heatmap-label"></div>']
        for j in range(world_size):
            header_cells.append(f'<div class="heatmap-label">R{j}</div>')
        rows.append(f'<div class="heatmap grid-{world_size}">{"".join(header_cells)}</div>')

        # 数据行
        data_rows = []
        for i in range(world_size):
            cells = [f'<div class="heatmap-label">R{i}</div>']
            for j in range(world_size):
                if i == j:
                    cells.append('<div class="heatmap-cell diagonal">-</div>')
                else:
                    val = matrix_2d[i][j]
                    link = self.matrix.get_link(i, j)

                    # 计算颜色
                    if link:
                        if link.is_slow_link:
                            color = self.COLORS["slow_link"]
                            text_color = "white"
                        elif link.is_bottleneck:
                            color = self.COLORS["bottleneck"]
                            text_color = "white"
                        else:
                            # 根据值计算颜色渐变
                            ratio = val / max_val if max_val > 0 else 0
                            color = self._get_color_for_ratio(ratio)
                            text_color = "white" if ratio > 0.5 else "#333"

                        # 格式化显示值
                        if data_type == "bandwidth":
                            display_val = f"{val:.1f}"
                        else:
                            display_val = f"{val*100:.0f}%"

                        # 构建悬停信息
                        info = self._build_cell_info(link, val)

                        cells.append(
                            f'<div class="heatmap-cell" style="background: {color}; color: {text_color}" '
                            f'data-info="{info}">{display_val}</div>'
                        )
                    else:
                        cells.append('<div class="heatmap-cell" style="background: #f5f5f5; color: #999">-</div>')

            data_rows.append(f'<div class="heatmap grid-{world_size}">{"".join(cells)}</div>')

        return "\n".join(data_rows)

    def _get_color_for_ratio(self, ratio: float) -> str:
        """根据比例获取颜色"""
        if ratio < 0.3:
            # 绿色
            return f"rgb(76, 175, 80)"
        elif ratio < 0.6:
            # 黄色
            r = int(76 + (255 - 76) * (ratio - 0.3) / 0.3)
            g = int(175 + (193 - 175) * (ratio - 0.3) / 0.3)
            b = int(80 + (7 - 80) * (ratio - 0.3) / 0.3)
            return f"rgb({r}, {g}, {b})"
        else:
            # 红色
            r = int(255)
            g = int(193 - 193 * (ratio - 0.6) / 0.4)
            b = int(7 - 7 * (ratio - 0.6) / 0.4)
            return f"rgb({r}, {max(0, g)}, {max(0, b)})"

    def _build_cell_info(self, link: LinkMetrics, val: float) -> str:
        """构建单元格悬停信息"""
        info_parts = [
            f"Rank {link.src_rank} → {link.dst_rank}",
            f"带宽: {val:.2f} GB/s",
            f"类型: {link.transport_type.value.upper()}",
        ]
        if link.bandwidth_utilization:
            info_parts.append(f"利用率: {link.bandwidth_utilization*100:.1f}%")
        if link.total_transit_size_mb:
            info_parts.append(f"数据量: {link.total_transit_size_mb:.2f} MB")
        return " | ".join(info_parts)

    def _generate_link_table(self) -> str:
        """生成链路详情表格"""
        if not self.matrix.link_metrics:
            return "<p>无链路数据</p>"

        # 按带宽排序
        sorted_links = sorted(
            self.matrix.link_metrics.values(),
            key=lambda x: x.achieved_bandwidth_gbps or 0,
            reverse=True
        )

        rows = []
        for link in sorted_links[:50]:  # 最多显示 50 条
            transport_badge = f'<span class="badge {link.transport_type.value}">{link.transport_type.value.upper()}</span>'

            status_badge = ""
            if link.is_slow_link:
                status_badge = ' <span class="badge slow">慢链路</span>'
            elif link.is_bottleneck:
                status_badge = ' <span class="badge bottleneck">瓶颈</span>'

            rows.append(f"""
                <tr>
                    <td>R{link.src_rank} ↔ R{link.dst_rank}</td>
                    <td>{transport_badge}{status_badge}</td>
                    <td>{link.achieved_bandwidth_gbps:.2f} GB/s</td>
                    <td>{link.bandwidth_utilization*100:.1f}%</td>
                    <td>{link.total_transit_size_mb:.2f} MB</td>
                    <td>{link.total_transit_time_ms:.2f} ms</td>
                    <td>{link.op_count}</td>
                </tr>
            """)

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>链路</th>
                        <th>类型</th>
                        <th>带宽</th>
                        <th>利用率</th>
                        <th>数据量</th>
                        <th>时间</th>
                        <th>操作数</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _generate_slow_links_section(self) -> str:
        """生成慢链路警告区域"""
        slow_links = self.matrix.slow_links

        if not slow_links:
            return ""

        items = []
        for link in slow_links[:10]:
            items.append(f"""
                <li>
                    <strong>R{link.src_rank} ↔ R{link.dst_rank}</strong>:
                    带宽 {link.achieved_bandwidth_gbps:.2f} GB/s
                    ({link.bandwidth_utilization*100:.1f}% 利用率)
                </li>
            """)

        return f"""
            <div class="card">
                <div class="alert alert-warning">
                    <h4>⚠️ 检测到 {len(slow_links)} 条慢链路</h4>
                    <p>以下链路的带宽利用率显著低于平均值，可能影响训练性能：</p>
                    <ul>
                        {"".join(items)}
                    </ul>
                </div>
            </div>
        """

    def _generate_summary_section(self) -> str:
        """生成统计摘要区域"""
        return f"""
            <div class="card">
                <h2>📈 通信概览</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{self.matrix.total_comm_data_mb:.1f}</div>
                        <div class="metric-label">总通信量 (MB)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.matrix.total_comm_time_ms:.1f}</div>
                        <div class="metric-label">总通信时间 (ms)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.matrix.avg_bandwidth_gbps:.1f}</div>
                        <div class="metric-label">平均带宽 (GB/s)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(self.matrix.link_metrics)}</div>
                        <div class="metric-label">活跃链路数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(self.matrix.slow_links)}</div>
                        <div class="metric-label">慢链路数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(self.matrix.bottleneck_links)}</div>
                        <div class="metric-label">瓶颈链路数</div>
                    </div>
                </div>
            </div>
        """


def visualize_communication_matrix(
    matrix: CommunicationMatrix,
    output_path: Optional[str] = None,
    title: str = "通信矩阵分析",
) -> str:
    """
    便捷函数：生成通信矩阵可视化

    Args:
        matrix: CommunicationMatrix 对象
        output_path: 输出路径（可选）
        title: 页面标题

    Returns:
        HTML 内容字符串
    """
    visualizer = CommunicationMatrixVisualizer(matrix)
    return visualizer.generate_html(title=title, output_path=output_path)
