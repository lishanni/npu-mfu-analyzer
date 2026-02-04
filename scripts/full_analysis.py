#!/usr/bin/env python3
"""
完整分析 Profiling 数据并生成详细报告
不依赖 LLM，使用统计分析方法
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import ProfilingLoader
from src.hardware import detect_hardware
from src.data_loader.data_validator import validate_and_parse
from src.data_loader.stream_parser import StreamParser, TimelineSummarizer
from src.analyzers import MFUCalculator


def find_trace_view(profiling_dir: Path) -> Path:
    """查找 trace_view.json 文件"""
    possible_paths = [
        profiling_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json",
        profiling_dir / "ASCEND_PROFILER_OUTPUT" / "1" / "trace_view.json",
        profiling_dir / "trace_view.json",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # 尝试递归查找
    for path in profiling_dir.rglob("trace_view.json"):
        return path

    return None


def analyze_memory_stats(loader: ProfilingLoader) -> dict:
    """分析内存统计"""
    try:
        mem_stats = loader.get_memory_stats()
        if mem_stats is not None and not mem_stats.empty:
            return {
                "peak_memory_mb": mem_stats.get("peak_memory_mb", 0),
                "memory_utilization": mem_stats.get("memory_utilization", 0),
            }
    except Exception as e:
        pass
    return {}


def analyze_communication(loader: ProfilingLoader) -> dict:
    """分析通信统计"""
    comm_stats = {
        "total_comm_time_ms": 0,
        "comm_ratio": 0,
        "collective_ops_count": 0,
        "collective_ops_details": defaultdict(list),
    }

    try:
        # 从 step trace 获取通信信息
        step_trace = loader.get_step_trace()
        if step_trace is not None and not step_trace.empty:
            # 检查通信相关列
            comm_cols = [col for col in step_trace.columns if "comm" in col.lower()]

            for col in comm_cols:
                comm_stats["total_comm_time_ms"] += step_trace[col].sum()

            # 计算通信占比
            total_time = step_trace["duration"].sum() if "duration" in step_trace.columns else 0
            if total_time > 0:
                comm_stats["comm_ratio"] = comm_stats["total_comm_time_ms"] / total_time
    except Exception as e:
        pass

    return comm_stats


def calculate_mfu(hw_spec, loader: ProfilingLoader) -> dict:
    """计算 MFU"""
    mfu_stats = {
        "compute_mfu": 0,
        "memory_mfu": 0,
        "overall_mfu": 0,
        "achieved_tflops": 0,
    }

    try:
        calculator = MFUCalculator(hw_spec)

        step_trace = loader.get_step_trace()
        if step_trace is not None and not step_trace.empty:
            # 获取计算时间
            computing_time_ms = 0
            if "computing" in step_trace.columns:
                computing_time_ms = step_trace["computing"].sum()

            if computing_time_ms > 0:
                # 假设模型参数（需要用户提供或从配置读取）
                # 这里使用默认值进行估算
                model_params = 7e9  # 7B 模型
                seq_len = 2048
                batch_size = 1
                num_ranks = 1

                # 计算实现的 FLOPS
                # 这是简化计算，实际情况需要根据具体模型类型
                total_flops = calculator.estimate_model_flops(model_params, seq_len, batch_size)

                # 计算实际 TFLOPS
                total_time_sec = computing_time_ms / 1000 / len(step_trace)
                achieved_tflops = (total_flops * num_ranks) / total_time_sec / 1e12

                # 计算 MFU
                peak_tflops = hw_spec.peak_tflops_fp16 * num_ranks * hw_spec.aicore_count
                mfu_stats["compute_mfu"] = achieved_tflops / peak_tflops if peak_tflops > 0 else 0
                mfu_stats["achieved_tflops"] = achieved_tflops

            # 内存 MFU（简化计算）
            if "hbm_bandwidth_gbps" in hw_spec.__dict__:
                # 假设内存访问量
                memory_access_gb = model_params * 4 / 1e9  # FP16 参数
                achieved_bandwidth = memory_access_gb / total_time_sec if total_time_sec > 0 else 0
                mfu_stats["memory_mfu"] = achieved_bandwidth / hw_spec.hbm_bandwidth_gbps if hw_spec.hbm_bandwidth_gbps > 0 else 0

            # 整体 MFU（取较小值）
            mfu_stats["overall_mfu"] = min(mfu_stats["compute_mfu"], mfu_stats["memory_mfu"])

    except Exception as e:
        mfu_stats["error"] = str(e)

    return mfu_stats


def generate_full_report(profiling_path: str, output_path: str):
    """生成完整的分析报告"""

    profiling_dir = Path(profiling_path)
    report_lines = []

    # 标题
    report_lines.append("# NPU MFU 完整分析报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Profiling 路径**: {profiling_dir}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    print(f"开始分析 Profiling 数据: {profiling_dir}")
    print("=" * 60)

    # 1. 硬件信息
    print("\n[1/7] 硬件检测...")
    report_lines.append("## 1. 硬件信息")
    report_lines.append("")

    hw_spec = detect_hardware(str(profiling_dir))
    if hw_spec:
        report_lines.append(f"- **芯片型号**: {hw_spec.name} ({hw_spec.variant})")
        report_lines.append(f"- **AICore 数量**: {hw_spec.aicore_count}")
        report_lines.append(f"- **AICore 频率**: {hw_spec.aicore_freq_mhz} MHz")
        report_lines.append(f"- **FP16 峰值算力**: {hw_spec.peak_tflops_fp16} TFLOPS")
        report_lines.append(f"- **HBM 带宽**: {hw_spec.hbm_bandwidth_gbps} GB/s")
        report_lines.append(f"- **HBM 容量**: {hw_spec.hbm_capacity_gb} GB")
        report_lines.append(f"- **HCCS 带宽**: {hw_spec.hccs_bandwidth_gbps} GB/s (×{hw_spec.hccs_links})")
        print(f"  检测到: {hw_spec.name} ({hw_spec.variant})")
    else:
        report_lines.append("- 未检测到硬件信息")
    report_lines.append("")

    # 2. 数据质量检查
    print("[2/7] 数据质量检查...")
    report_lines.append("## 2. 数据质量检查")
    report_lines.append("")

    trace_view_path = find_trace_view(profiling_dir)
    if trace_view_path:
        print(f"  找到 trace_view.json: {trace_view_path}")

        # 使用 TimelineSummarizer 进行摘要
        try:
            summary = TimelineSummarizer.summarize_file(str(trace_view_path), max_top_events=50)

            report_lines.append(f"- **总事件数**: {summary['total_events']}")
            report_lines.append(f"- **总持续时间**: {summary['total_duration_us'] / 1000:.2f} ms")

            # 按分类统计
            report_lines.append("")
            report_lines.append("### 事件分类统计")
            report_lines.append("")
            report_lines.append("| Category | 事件数 | 总耗时 (ms) |")
            report_lines.append("|----------|--------|-------------|")

            for cat, stats in sorted(summary['by_category'].items(), key=lambda x: x[1]['duration'], reverse=True):
                cat_name = cat if cat else "未知"
                count = stats['count']
                duration_ms = stats['duration'] / 1000
                report_lines.append(f"| {cat_name} | {count} | {duration_ms:.2f} |")

            # Top 事件
            report_lines.append("")
            report_lines.append("### Top 20 耗时事件")
            report_lines.append("")
            report_lines.append("| 排名 | 事件名称 | Category | 耗时 (μs) |")
            report_lines.append("|------|----------|----------|-----------|")

            for i, (dur, name, cat) in enumerate(summary['top_by_duration'][:20], 1):
                report_lines.append(f"| {i} | {name[:50]} | {cat} | {dur:.0f} |")

            print(f"  总事件数: {summary['total_events']}")
            print(f"  持续时间: {summary['total_duration_us'] / 1000:.2f} ms")

        except Exception as e:
            report_lines.append(f"Timeline 分析失败: {e}")
            print(f"  Timeline 分析失败: {e}")

        # 数据验证
        try:
            events, quality_report = validate_and_parse(str(trace_view_path))
            report_lines.append("")
            report_lines.append("### 数据质量")
            report_lines.append("")
            report_lines.append(f"- **质量等级**: {quality_report.quality_level.value.upper()}")
            report_lines.append(f"- **有效事件**: {quality_report.valid_events} / {quality_report.total_events}")

            if quality_report.error_count > 0 or quality_report.warning_count > 0:
                report_lines.append("")
                report_lines.append("#### 发现的问题")
                for issue in quality_report.issues[:10]:
                    report_lines.append(f"- **{issue.issue_type.value}** ({issue.severity}): {issue.message}")
        except Exception as e:
            report_lines.append(f"数据验证失败: {e}")
    else:
        report_lines.append(f"- 未找到 trace_view.json 文件")
    report_lines.append("")

    # 3. Profiling 数据概览
    print("[3/7] 加载 Profiling 数据...")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 3. Profiling 数据概览")
    report_lines.append("")

    try:
        loader = ProfilingLoader(profiling_dir)

        # Step trace
        step_trace = loader.get_step_trace()
        if step_trace is not None and not step_trace.empty:
            report_lines.append("### Step Trace 分析")
            report_lines.append("")
            report_lines.append(f"- **Step 数量**: {len(step_trace)}")

            # 显示所有列
            report_lines.append("")
            report_lines.append("**数据列**: " + ", ".join(step_trace.columns.tolist()))
            report_lines.append("")

            # 计算时间统计
            numeric_cols = step_trace.select_dtypes(include=['number']).columns.tolist()

            report_lines.append("### 时间统计")
            report_lines.append("")
            report_lines.append("| 指标 | " + " | ".join(numeric_cols[:10]) + " |")
            report_lines.append("|------|" + "|".join(["--------" for _ in numeric_cols[:10]]) + "|")
            report_lines.append("| 平均 | " + " | ".join([f"{step_trace[col].mean():.2f}" for col in numeric_cols[:10]]) + " |")
            report_lines.append("| 最小 | " + " | ".join([f"{step_trace[col].min():.2f}" for col in numeric_cols[:10]]) + " |")
            report_lines.append("| 最大 | " + " | ".join([f"{step_trace[col].max():.2f}" for col in numeric_cols[:10]]) + " |")

            print(f"  Step 数量: {len(step_trace)}")

        # Top 算子
        print("[4/7] 分析 Top 算子...")
        report_lines.append("")
        report_lines.append("### Top 耗时算子 (Top 20)")
        report_lines.append("")

        try:
            top_kernels = loader.get_top_kernels(rank=None, top_n=20)
            if top_kernels:
                report_lines.append("| 排名 | 算子名称 | 总耗时 (μs) | 调用次数 | 平均耗时 (μs) | 类型 |")
                report_lines.append("|------|----------|-------------|----------|---------------|------|")
                for i, kernel in enumerate(top_kernels[:20], 1):
                    name = kernel.get('name', 'N/A')[:40]
                    total_dur = kernel.get('total_dur_us', 0)
                    count = kernel.get('count', 0)
                    avg_dur = kernel.get('avg_dur_us', 0)
                    op_type = kernel.get('type', 'N/A')
                    report_lines.append(f"| {i} | {name} | {total_dur:.2f} | {count} | {avg_dur:.2f} | {op_type} |")
                print(f"  Top 算子数量: {len(top_kernels)}")
            else:
                report_lines.append("无算子数据")
        except Exception as e:
            report_lines.append(f"获取算子数据失败: {e}")
            print(f"  获取算子数据失败: {e}")
        report_lines.append("")

    except Exception as e:
        report_lines.append(f"加载 Profiling 数据时出现错误: {e}")
        import traceback
        report_lines.append(f"```\n{traceback.format_exc()}\n```")
        print(f"  加载错误: {e}")

    # 4. MFU 分析
    print("[5/7] 计算 MFU...")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. MFU (Model FLOPS Utilization) 分析")
    report_lines.append("")

    if hw_spec:
        try:
            mfu_stats = calculate_mfu(hw_spec, loader)
            report_lines.append(f"- **计算 MFU**: {mfu_stats.get('compute_mfu', 0)*100:.2f}%")
            report_lines.append(f"- **内存 MFU**: {mfu_stats.get('memory_mfu', 0)*100:.2f}%")
            report_lines.append(f"- **整体 MFU**: {mfu_stats.get('overall_mfu', 0)*100:.2f}%")
            report_lines.append(f"- **实现算力**: {mfu_stats.get('achieved_tflops', 0):.2f} TFLOPS")
            report_lines.append("")
            print(f"  整体 MFU: {mfu_stats.get('overall_mfu', 0)*100:.2f}%")
        except Exception as e:
            report_lines.append(f"MFU 计算失败: {e}")
            print(f"  MFU 计算失败: {e}")

    # 5. 通信分析
    print("[6/7] 通信分析...")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 5. 通信分析")
    report_lines.append("")

    try:
        comm_stats = analyze_communication(loader)
        report_lines.append(f"- **通信总耗时**: {comm_stats['total_comm_time_ms']:.2f} ms")
        report_lines.append(f"- **通信占比**: {comm_stats['comm_ratio']*100:.1f}%")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"通信分析失败: {e}")

    # 6. 优化建议
    print("[7/7] 生成优化建议...")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 6. 优化建议")
    report_lines.append("")

    # 基于分析结果生成建议
    recommendations = []

    # 检查空闲时间
    if step_trace is not None and "free" in step_trace.columns:
        free_ratio = (step_trace["free"].sum() / step_trace["duration"].sum()) if "duration" in step_trace.columns else 0
        if free_ratio > 0.3:
            recommendations.append("1. **减少空闲时间**：空闲时间占比超过 30%，建议检查数据加载、内存分配等瓶颈")

    # 检查通信掩盖
    if comm_stats.get("comm_ratio", 0) > 0.1:
        recommendations.append("2. **优化通信掩盖**：通信时间占比较高，建议使用梯度累积、通信融合等技术")

    # 检查 MFU
    if hw_spec:
        mfu = mfu_stats.get("overall_mfu", 0)
        if mfu < 0.3:
            recommendations.append("3. **提高算力利用率**：MFU 低于 30%，建议优化 Kernel 实现、使用混合精度训练")
        elif mfu < 0.5:
            recommendations.append("3. **提高算力利用率**：MFU 低于 50%，建议优化模型并行策略、减少冗余计算")

    if not recommendations:
        recommendations.append("基于当前分析未发现明显性能瓶颈。建议：")
        recommendations.append("1. 持续监控 MFU 和通信掩盖率")
        recommendations.append("2. 关注算子融合和内存复用")
        recommendations.append("3. 根据实际情况调整并行策略")

    for rec in recommendations:
        report_lines.append(rec)
    report_lines.append("")

    # 写入报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n报告已生成: {output_file}")
    print(f"报告大小: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)

    return "\n".join(report_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python scripts/full_analysis.py <profiling_path> [output_path]")
        sys.exit(1)

    profiling_path = sys.argv[1]

    # 默认输出到上级目录
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = str(Path(__file__).parent.parent.parent / f"npu_full_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    generate_full_report(profiling_path, output_path)
