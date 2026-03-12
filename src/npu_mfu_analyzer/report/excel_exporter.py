"""
Excel 报告导出

将分析结果导出为 Excel 格式。
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ExcelExporter:
    """
    Excel 报告导出器
    
    功能：
    1. 导出性能概览
    2. 导出时间分布
    3. 导出优化建议
    4. 导出历史对比
    5. 支持图表
    """
    
    def __init__(self):
        self._workbook = None
        self._formats = {}
    
    def export(
        self,
        output_path: str,
        profiling_summary: Any,
        agent_results: Dict[str, Any] = None,
        suggestions: List[Dict[str, Any]] = None,
        comparison: Any = None,
    ):
        """
        导出 Excel 报告
        
        Args:
            output_path: 输出文件路径
            profiling_summary: ProfilingSummary 对象
            agent_results: Agent 分析结果
            suggestions: 优化建议列表
            comparison: 历史对比结果
        """
        try:
            import xlsxwriter
        except ImportError:
            raise ImportError("xlsxwriter not installed. Run: pip install xlsxwriter")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._workbook = xlsxwriter.Workbook(str(output_path))
        self._init_formats()
        
        try:
            # 1. 概览页
            self._write_overview_sheet(profiling_summary)
            
            # 2. 时间分布页
            self._write_time_distribution_sheet(profiling_summary)
            
            # 3. 优化建议页
            if suggestions:
                self._write_suggestions_sheet(suggestions)
            
            # 4. Agent 分析结果页
            if agent_results:
                self._write_agent_results_sheet(agent_results)
            
            # 5. 历史对比页
            if comparison:
                self._write_comparison_sheet(comparison)
            
            logger.info(f"Excel report exported to: {output_path}")
            
        finally:
            self._workbook.close()
    
    def _init_formats(self):
        """初始化格式"""
        self._formats = {
            "title": self._workbook.add_format({
                "bold": True,
                "font_size": 16,
                "font_color": "#1a73e8",
            }),
            "header": self._workbook.add_format({
                "bold": True,
                "bg_color": "#1a73e8",
                "font_color": "white",
                "border": 1,
            }),
            "cell": self._workbook.add_format({
                "border": 1,
            }),
            "number": self._workbook.add_format({
                "border": 1,
                "num_format": "#,##0.00",
            }),
            "percent": self._workbook.add_format({
                "border": 1,
                "num_format": "0.0%",
            }),
            "good": self._workbook.add_format({
                "border": 1,
                "bg_color": "#e6f4ea",
                "font_color": "#137333",
            }),
            "warning": self._workbook.add_format({
                "border": 1,
                "bg_color": "#fef7e0",
                "font_color": "#b45309",
            }),
            "bad": self._workbook.add_format({
                "border": 1,
                "bg_color": "#fce8e6",
                "font_color": "#c5221f",
            }),
        }
    
    def _write_overview_sheet(self, summary: Any):
        """写入概览页"""
        sheet = self._workbook.add_worksheet("概览")
        
        # 标题
        sheet.write(0, 0, "昇腾 NPU 性能分析报告", self._formats["title"])
        sheet.write(1, 0, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 提取数据
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else {}
        
        # 核心指标表格
        sheet.write(3, 0, "核心指标", self._formats["header"])
        sheet.write(3, 1, "数值", self._formats["header"])
        
        metrics = [
            ("Step 数量", summary_dict.get("step_count", 0)),
            ("平均 Step 时间 (ms)", summary_dict.get("avg_step_time", 0) / 1000),
            ("Rank 数量", summary_dict.get("rank_count", 1)),
        ]
        
        for i, (name, value) in enumerate(metrics, 4):
            sheet.write(i, 0, name, self._formats["cell"])
            sheet.write(i, 1, value, self._formats["number"])
        
        # 设置列宽
        sheet.set_column(0, 0, 25)
        sheet.set_column(1, 1, 15)
    
    def _write_time_distribution_sheet(self, summary: Any):
        """写入时间分布页"""
        sheet = self._workbook.add_worksheet("时间分布")
        
        # 提取数据
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else {}
        
        compute_time = summary_dict.get("avg_compute_time", 0)
        comm_time = summary_dict.get("avg_comm_time", 0)
        free_time = summary_dict.get("avg_free_time", 0)
        total_time = compute_time + comm_time + free_time
        
        # 表格
        sheet.write(0, 0, "类别", self._formats["header"])
        sheet.write(0, 1, "时间 (us)", self._formats["header"])
        sheet.write(0, 2, "占比", self._formats["header"])
        
        data = [
            ("计算", compute_time, compute_time / total_time if total_time > 0 else 0),
            ("通信", comm_time, comm_time / total_time if total_time > 0 else 0),
            ("空闲", free_time, free_time / total_time if total_time > 0 else 0),
            ("总计", total_time, 1.0),
        ]
        
        for i, (name, time_us, ratio) in enumerate(data, 1):
            sheet.write(i, 0, name, self._formats["cell"])
            sheet.write(i, 1, time_us, self._formats["number"])
            sheet.write(i, 2, ratio, self._formats["percent"])
        
        # 添加饼图
        chart = self._workbook.add_chart({"type": "pie"})
        chart.add_series({
            "name": "时间分布",
            "categories": f"='时间分布'!$A$2:$A$4",
            "values": f"='时间分布'!$C$2:$C$4",
            "data_labels": {"percentage": True},
        })
        chart.set_title({"name": "时间分布"})
        chart.set_size({"width": 400, "height": 300})
        
        sheet.insert_chart("E2", chart)
        
        # 设置列宽
        sheet.set_column(0, 0, 15)
        sheet.set_column(1, 2, 15)
    
    def _write_suggestions_sheet(self, suggestions: List[Dict[str, Any]]):
        """写入优化建议页"""
        sheet = self._workbook.add_worksheet("优化建议")
        
        # 表头
        headers = ["优先级", "标题", "描述", "预期收益"]
        for col, header in enumerate(headers):
            sheet.write(0, col, header, self._formats["header"])
        
        # 数据
        priority_format = {
            "high": self._formats["bad"],
            "medium": self._formats["warning"],
            "low": self._formats["good"],
        }
        
        for row, s in enumerate(suggestions, 1):
            priority = s.get("priority", "medium")
            fmt = priority_format.get(priority, self._formats["cell"])
            
            sheet.write(row, 0, priority.upper(), fmt)
            sheet.write(row, 1, s.get("title", ""), self._formats["cell"])
            sheet.write(row, 2, s.get("description", ""), self._formats["cell"])
            sheet.write(row, 3, s.get("expected_benefit", ""), self._formats["cell"])
        
        # 设置列宽
        sheet.set_column(0, 0, 10)
        sheet.set_column(1, 1, 40)
        sheet.set_column(2, 2, 50)
        sheet.set_column(3, 3, 20)
    
    def _write_agent_results_sheet(self, agent_results: Dict[str, Any]):
        """写入 Agent 分析结果页"""
        sheet = self._workbook.add_worksheet("Agent分析")
        
        row = 0
        for name, result in agent_results.items():
            sheet.write(row, 0, name, self._formats["header"])
            row += 1
            
            if hasattr(result, "summary"):
                sheet.write(row, 0, "摘要", self._formats["cell"])
                sheet.write(row, 1, result.summary, self._formats["cell"])
                row += 1
            
            if hasattr(result, "details") and result.details:
                for key, value in result.details.items():
                    sheet.write(row, 0, key, self._formats["cell"])
                    sheet.write(row, 1, str(value), self._formats["cell"])
                    row += 1
            
            row += 1  # 空行
        
        sheet.set_column(0, 0, 25)
        sheet.set_column(1, 1, 60)
    
    def _write_comparison_sheet(self, comparison: Any):
        """写入历史对比页"""
        sheet = self._workbook.add_worksheet("历史对比")
        
        # 标题
        sheet.write(0, 0, "性能对比", self._formats["title"])
        
        if hasattr(comparison, "baseline") and hasattr(comparison, "current"):
            # 表头
            sheet.write(2, 0, "指标", self._formats["header"])
            sheet.write(2, 1, "基准值", self._formats["header"])
            sheet.write(2, 2, "当前值", self._formats["header"])
            sheet.write(2, 3, "变化", self._formats["header"])
            
            baseline = comparison.baseline
            current = comparison.current
            
            data = [
                ("Step 时间 (ms)", baseline.avg_step_time_us/1000, current.avg_step_time_us/1000, comparison.step_time_change_pct),
                ("估算 MFU", baseline.estimated_mfu, current.estimated_mfu, comparison.mfu_change_pct),
                ("计算占比", baseline.compute_ratio, current.compute_ratio, comparison.compute_change_pct),
                ("通信占比", baseline.comm_ratio, current.comm_ratio, comparison.comm_change_pct),
                ("空闲占比", baseline.idle_ratio, current.idle_ratio, comparison.idle_change_pct),
            ]
            
            for row, (name, base_val, curr_val, change) in enumerate(data, 3):
                sheet.write(row, 0, name, self._formats["cell"])
                sheet.write(row, 1, base_val, self._formats["number"])
                sheet.write(row, 2, curr_val, self._formats["number"])
                
                # 根据变化选择格式
                if abs(change) < 1:
                    fmt = self._formats["cell"]
                elif (name in ["Step 时间 (ms)", "通信占比", "空闲占比"] and change > 0) or \
                     (name in ["估算 MFU", "计算占比"] and change < 0):
                    fmt = self._formats["bad"]
                else:
                    fmt = self._formats["good"]
                
                sheet.write(row, 3, f"{change:+.1f}%", fmt)
        
        sheet.set_column(0, 0, 20)
        sheet.set_column(1, 3, 15)
