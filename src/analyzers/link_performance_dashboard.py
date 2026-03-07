"""
链路性能仪表板模块

生成功能完整的 HTML 链路性能仪表板，包括：
- 实时指标卡片
- 交互式热力图
- 趋势图表
- 异常链路分析
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrix,
    LinkMetrics,
    TransportType,
)

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """仪表板数据结构"""
    summary: Dict[str, Any]
    links: List[Dict[str, Any]]
    heatmap: Dict[str, Any]
    distributions: Dict[str, Any]
    anomalies: Dict[str, Any]


class LinkPerformanceDashboard:
    """
    链路性能仪表板生成器

    生成功能完整的 HTML 仪表板，支持实时指标卡片、交互式热力图、
    趋势图表、异常链路分析等功能。
    """

    # 颜色配置
    COLORS = {
        "hccs": "#4CAF50",      # 绿色 - 节点内
        "rdma": "#2196F3",       # 蓝色 - 跨节点
        "roce": "#9C27B0",       # 紫色 - RoCE
        "pcie": "#FF9800",       # 橙色 - PCIe
        "slow_link": "#F44336",  # 红色 - 慢链路
        "bottleneck": "#E91E63", # 粉色 - 瓶颈链路
        "primary": "#1a73e8",    # 主色
        "success": "#34a853",    # 成功色
        "warning": "#fbbc04",    # 警告色
        "danger": "#ea4335",     # 危险色
    }

    def __init__(self, matrix: CommunicationMatrix):
        """
        Args:
            matrix: CommunicationMatrix 对象
        """
        self.matrix = matrix
        self.data = self._prepare_data()

    def _prepare_data(self) -> DashboardData:
        """准备图表数据"""
        return DashboardData(
            summary=self._get_summary_metrics(),
            links=self._get_link_data(),
            heatmap=self._get_heatmap_data(),
            distributions=self._get_distributions(),
            anomalies=self._get_anomalies(),
        )

    def _get_summary_metrics(self) -> Dict[str, Any]:
        """获取摘要指标"""
        matrix = self.matrix

        # 计算通信效率评分
        efficiency_score = self._calculate_efficiency_score()

        # 获取慢链路和瓶颈链路数量
        slow_count = len(matrix.slow_links)
        bottleneck_count = len(matrix.bottleneck_links)

        return {
            "world_size": matrix.world_size,
            "total_comm_data_mb": round(matrix.total_comm_data_mb, 2),
            "total_comm_time_ms": round(matrix.total_comm_time_ms, 2),
            "avg_bandwidth_gbps": round(matrix.avg_bandwidth_gbps, 2),
            "peak_bandwidth_gbps": round(matrix.peak_bandwidth_gbps, 2),
            "intra_node_ratio": round(matrix.intra_node_ratio, 3),
            "inter_node_ratio": round(matrix.inter_node_ratio, 3),
            "hccs_ratio": round(matrix.hccs_ratio, 3),
            "rdma_ratio": round(matrix.rdma_ratio, 3),
            "slow_link_count": slow_count,
            "bottleneck_link_count": bottleneck_count,
            "efficiency_score": round(efficiency_score, 2),
        }

    def _calculate_efficiency_score(self) -> float:
        """计算通信效率评分 (0-100)"""
        if not self.matrix.link_metrics:
            return 0.0

        # 基于带宽利用率和慢链路比例计算
        avg_util = sum(
            m.bandwidth_utilization for m in self.matrix.link_metrics.values()
        ) / len(self.matrix.link_metrics)

        # 慢链路惩罚
        slow_ratio = len(self.matrix.slow_links) / len(self.matrix.link_metrics)
        slow_penalty = slow_ratio * 30

        # 效率评分 = 平均利用率 * 100 - 慢链路惩罚
        score = max(0, min(100, avg_util * 100 - slow_penalty))
        return score

    def _get_link_data(self) -> List[Dict[str, Any]]:
        """获取链路数据"""
        links = []
        for metrics in self.matrix.link_metrics.values():
            links.append({
                "src_rank": metrics.src_rank,
                "dst_rank": metrics.dst_rank,
                "transport_type": metrics.transport_type.value,
                "theoretical_bandwidth_gbps": round(metrics.theoretical_bandwidth_gbps, 2),
                "achieved_bandwidth_gbps": round(metrics.achieved_bandwidth_gbps, 2),
                "bandwidth_utilization": round(metrics.bandwidth_utilization, 3),
                "total_transit_size_mb": round(metrics.total_transit_size_mb, 2),
                "total_transit_time_ms": round(metrics.total_transit_time_ms, 2),
                "avg_latency_us": round(metrics.avg_latency_us, 2),
                "op_count": metrics.op_count,
                "is_slow_link": metrics.is_slow_link,
                "is_bottleneck": metrics.is_bottleneck,
                "anomaly_score": round(metrics.anomaly_score, 3),
            })
        return links

    def _get_heatmap_data(self) -> Dict[str, Any]:
        """获取热力图数据"""
        world_size = self.matrix.world_size

        # 带宽矩阵
        bandwidth_matrix = self.matrix.get_matrix_2d()

        # 利用率矩阵
        utilization_matrix = self.matrix.get_utilization_matrix_2d()

        # 传输类型矩阵
        transport_matrix = [[None] * world_size for _ in range(world_size)]
        for (src, dst), metrics in self.matrix.link_metrics.items():
            transport_matrix[src][dst] = metrics.transport_type.value
            transport_matrix[dst][src] = metrics.transport_type.value

        return {
            "bandwidth_matrix": bandwidth_matrix,
            "utilization_matrix": utilization_matrix,
            "transport_matrix": transport_matrix,
            "world_size": world_size,
        }

    def _get_distributions(self) -> Dict[str, Any]:
        """获取分布数据"""
        links = list(self.matrix.link_metrics.values())

        # 带直方图数据
        bandwidths = [m.achieved_bandwidth_gbps for m in links]
        bandwidth_bins = self._create_histogram(bandwidths, bins=20)

        # 利用率分布
        utilizations = [m.bandwidth_utilization * 100 for m in links]
        util_bins = self._create_histogram(utilizations, bins=10)

        # 延迟分布
        latencies = [m.avg_latency_us for m in links if m.avg_latency_us > 0]
        latency_stats = self._calculate_stats(latencies) if latencies else {}

        # 传输类型分布
        type_counts = defaultdict(int)
        type_sizes = defaultdict(float)
        for m in links:
            type_counts[m.transport_type.value] += 1
            type_sizes[m.transport_type.value] += m.total_transit_size_mb

        return {
            "bandwidth_histogram": bandwidth_bins,
            "utilization_distribution": util_bins,
            "latency_stats": latency_stats,
            "transport_type_counts": dict(type_counts),
            "transport_type_sizes": {k: round(v, 2) for k, v in type_sizes.items()},
        }

    def _create_histogram(self, values: List[float], bins: int = 10) -> List[Dict[str, Any]]:
        """创建直方图数据"""
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [{"label": f"{min_val:.2f}", "count": len(values)}]

        bin_width = (max_val - min_val) / bins
        histogram = []

        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for v in values if bin_start <= v < bin_end)

            histogram.append({
                "label": f"{bin_start:.1f}-{bin_end:.1f}",
                "count": count,
                "start": round(bin_start, 2),
                "end": round(bin_end, 2),
            })

        return histogram

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """计算统计数据"""
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(values)

        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "mean": round(sum(values) / n, 2),
            "median": round(sorted_values[n // 2], 2),
            "p50": round(sorted_values[int(n * 0.5)], 2) if n > 1 else sorted_values[0],
            "p95": round(sorted_values[int(n * 0.95)], 2) if n > 20 else sorted_values[-1],
            "p99": round(sorted_values[int(n * 0.99)], 2) if n > 100 else sorted_values[-1],
        }

    def _get_anomalies(self) -> Dict[str, Any]:
        """获取异常链路数据"""
        # 慢链路
        slow_links = [
            {
                "src_rank": m.src_rank,
                "dst_rank": m.dst_rank,
                "bandwidth_gbps": round(m.achieved_bandwidth_gbps, 2),
                "utilization": round(m.bandwidth_utilization, 3),
                "anomaly_score": round(m.anomaly_score, 3),
            }
            for m in self.matrix.slow_links[:20]
        ]

        # 瓶颈链路
        bottleneck_links = [
            {
                "src_rank": m.src_rank,
                "dst_rank": m.dst_rank,
                "bandwidth_gbps": round(m.achieved_bandwidth_gbps, 2),
                "utilization": round(m.bandwidth_utilization, 3),
                "total_size_mb": round(m.total_transit_size_mb, 2),
            }
            for m in self.matrix.bottleneck_links[:20]
        ]

        # 异常类型分类
        anomaly_types = defaultdict(int)
        for m in self.matrix.link_metrics.values():
            if m.is_slow_link:
                anomaly_types["slow"] += 1
            if m.is_bottleneck:
                anomaly_types["bottleneck"] += 1

        return {
            "slow_links": slow_links,
            "bottleneck_links": bottleneck_links,
            "anomaly_types": dict(anomaly_types),
        }

    def generate_html(
        self,
        output_path: Optional[str] = None,
        title: str = "链路性能仪表板",
    ) -> str:
        """
        生成 HTML 仪表板

        Args:
            output_path: 输出路径（可选）
            title: 页面标题

        Returns:
            HTML 内容字符串
        """
        # 加载模板
        template = self._load_template()

        # 嵌入数据
        data_json = json.dumps(self.data.__dict__, ensure_ascii=False)
        html = template.replace("/* DATA_PLACEHOLDER */", f"const DASHBOARD_DATA = {data_json};")
        html = html.replace("{{TITLE}}", title)

        # 保存
        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"Link performance dashboard saved to: {output_path}")

        return html

    def _load_template(self) -> str:
        """加载 HTML 模板"""
        template_path = Path(__file__).parent.parent / "web" / "static" / "dashboard" / "index.html"

        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        # 使用内嵌模板作为后备
        return self._get_embedded_template()

    def _get_embedded_template(self) -> str:
        """获取内嵌的 HTML 模板"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <link rel="stylesheet" href="css/dashboard.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🔗 链路性能仪表板</h1>
            <div class="meta">World Size: <span id="world-size">-</span></div>
        </header>

        <section class="metrics-section">
            <h2>📈 性能指标</h2>
            <div class="metrics-grid" id="metrics-grid">
                <!-- 指标卡片由 JS 生成 -->
            </div>
        </section>

        <section class="charts-section">
            <div class="chart-container">
                <h3>带宽分布</h3>
                <canvas id="bandwidth-chart"></canvas>
            </div>
            <div class="chart-container">
                <h3>利用率分布</h3>
                <canvas id="utilization-chart"></canvas>
            </div>
        </section>

        <section class="heatmap-section">
            <h2>🔥 通信矩阵热力图</h2>
            <div class="heatmap-controls">
                <select id="heatmap-type">
                    <option value="bandwidth">带宽 (GB/s)</option>
                    <option value="utilization">利用率 (%)</option>
                </select>
                <select id="transport-filter">
                    <option value="all">全部</option>
                    <option value="hccs">HCCS</option>
                    <option value="rdma">RDMA</option>
                </select>
            </div>
            <div id="heatmap-container"></div>
        </section>

        <section class="anomalies-section">
            <h2>⚠️ 异常链路</h2>
            <div class="anomaly-tabs">
                <button class="tab-btn active" data-tab="slow">慢链路</button>
                <button class="tab-btn" data-tab="bottleneck">瓶颈链路</button>
            </div>
            <div class="tab-content">
                <div id="slow-links-tab" class="tab-pane active"></div>
                <div id="bottleneck-links-tab" class="tab-pane"></div>
            </div>
        </section>
    </div>

    <script>
        /* DATA_PLACEHOLDER */
    </script>
    <script src="js/main.js"></script>
    <script src="js/charts.js"></script>
    <script src="js/heatmap.js"></script>
</body>
</html>"""


def generate_dashboard(
    matrix: CommunicationMatrix,
    output_path: Optional[str] = None,
    title: str = "链路性能仪表板",
) -> str:
    """
    便捷函数：生成链路性能仪表板

    Args:
        matrix: CommunicationMatrix 对象
        output_path: 输出路径（可选）
        title: 页面标题

    Returns:
        HTML 内容字符串
    """
    dashboard = LinkPerformanceDashboard(matrix)
    return dashboard.generate_html(output_path=output_path, title=title)
