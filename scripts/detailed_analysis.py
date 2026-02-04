#!/usr/bin/env python3
"""
详细分析 Profiling 数据并生成完整报告
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import ProfilingLoader
from src.hardware import detect_hardware
from src.data_loader.data_validator import validate_and_parse
from src.data_loader.stream_parser import StreamParser, TimelineSummarizer
import pandas as pd


def find_trace_view(profiling_dir: Path) -> Path:
    """查找 trace_view.json 文件"""
    # 尝试多个可能的位置
    possible_paths = [
        profiling_dir / "ASCEND_PROFILER_OUTPUT" / "trace_view.json",
        profiling_dir / "ASCEND_PROFILER_OUTPUT" / "1" / "trace_view.json",
        profiling_dir / "trace_view.json",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def analyze_profiling(profiling_path: str, output_path: str):
    """详细分析 Profiling 数据并生成报告"""

    profiling_dir = Path(profiling_path)
    report_lines = []

    # 标题
    report_lines.append("# NPU MFU 详细分析报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Profiling 路径**: {profiling_dir}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # 1. 硬件信息
    report_lines.append("## 1. 硬件信息")
    report_lines.append("")

    hw_spec = detect_hardware(str(profiling_dir))
    if hw_spec:
        report_lines.append(f"- **芯片型号**: {hw_spec.name} ({hw_spec.variant})")
        report_lines.append(f"- **AICore 数量**: {hw_spec.aicore_count}")
        report_lines.append(f"- **AICore 频率**: {hw_spec.aicore_freq_mhz} MHz")
        report_lines.append(f"- **FP16 峰值算力**: {hw_spec.peak_tflops_fp16} TFLOPS")
        report_lines.append(f"- **INT8 峰值算力**: {hw_spec.get_peak_tflops()*2:.0f} TFLOPS")
        report_lines.append(f"- **HBM 带宽**: {hw_spec.hbm_bandwidth_gbps} GB/s")
        report_lines.append(f"- **HBM 容量**: {hw_spec.hbm_capacity_gb} GB")
        report_lines.append(f"- **HCCS 带宽**: {hw_spec.hccs_bandwidth_gbps} GB/s (×{hw_spec.hccs_links})")
        report_lines.append(f"- **L2 Cache**: {hw_spec.l2_cache_mb} MB")
    else:
        report_lines.append("- 未检测到硬件信息")
    report_lines.append("")

    # 2. 数据质量检查
    report_lines.append("## 2. 数据质量检查")
    report_lines.append("")

    trace_view_path = find_trace_view(profiling_dir)
    if trace_view_path:
        print(f"找到 trace_view.json: {trace_view_path}")
        events, quality_report = validate_and_parse(str(trace_view_path))

        report_lines.append(f"- **质量等级**: {quality_report.quality_level.value.upper()}")
        report_lines.append(f"- **总事件数**: {quality_report.total_events}")
        report_lines.append(f"- **有效事件**: {quality_report.valid_events}")
        report_lines.append(f"- **跳过事件**: {quality_report.skipped_events}")
        report_lines.append(f"- **错误数**: {quality_report.error_count}")
        report_lines.append(f"- **警告数**: {quality_report.warning_count}")
        report_lines.append(f"- **有效率**: {quality_report.valid_events / quality_report.total_events * 100 if quality_report.total_events > 0 else 0:.1f}%")
        report_lines.append("")
        report_lines.append(f"- **包含 Timeline 数据**: {'是' if quality_report.has_timeline_data else '否'}")
        report_lines.append(f"- **包含 Kernel 数据**: {'是' if quality_report.has_kernel_data else '否'}")
        report_lines.append(f"- **包含通信数据**: {'是' if quality_report.has_communication_data else '否'}")

        if quality_report.issues and quality_report.total_issues > 0:
            report_lines.append("")
            report_lines.append("### 主要问题 (Top 10)")
            for issue in quality_report.issues[:10]:
                report_lines.append(f"- **{issue.issue_type.value}** ({issue.severity}): {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"  - 建议: {issue.suggestion}")
    else:
        report_lines.append(f"- ⚠️ trace_view.json 未找到")
    report_lines.append("")

    # 3. 使用 ProfilingLoader 加载数据
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

            # 计算时间统计
            time_cols = ['computing', 'communication', 'communication_not_overlapped',
                        'overlapped', 'free', 'bubble', 'stage']
            available_time_cols = [col for col in time_cols if col in step_trace.columns]

            if available_time_cols:
                total_time = step_trace[available_time_cols].sum(axis=1).mean()
                report_lines.append(f"- **平均总时间**: {total_time:.2f} ms")
                report_lines.append("")

                report_lines.append("#### 时间分布")
                report_lines.append("")
                for col in available_time_cols:
                    avg_val = step_trace[col].mean()
                    pct = (avg_val / total_time * 100) if total_time > 0 else 0
                    bar = "█" * int(pct / 5)
                    report_lines.append(f"- **{col}**: {avg_val:.2f} ms ({pct:.1f}%) {bar}")

            # 显示列名
            report_lines.append("")
            report_lines.append("**数据列**: " + ", ".join(step_trace.columns.tolist()))
            report_lines.append("")

        # Top 算子
        report_lines.append("### Top 耗时算子 (Top 20)")
        report_lines.append("")

        try:
            top_kernels = loader.get_top_kernels(rank=None, top_n=20)
            if top_kernels:
                report_lines.append("| 排名 | 算子名称 | 总耗时 (ms) | 调用次数 | 平均耗时 (μs) | 类型 |")
                report_lines.append("|------|----------|-------------|----------|---------------|------|")
                for i, kernel in enumerate(top_kernels[:20], 1):
                    name = kernel.get('name', 'N/A')
                    # 限制名称长度
                    if len(name) > 40:
                        name = name[:37] + "..."
                    total_dur_ms = kernel.get('total_dur_us', 0) / 1000
                    count = kernel.get('count', 0)
                    avg_dur_us = kernel.get('avg_dur_us', 0)
                    op_type = kernel.get('type', 'N/A')
                    report_lines.append(f"| {i} | {name} | {total_dur_ms:.2f} | {count} | {avg_dur_us:.2f} | {op_type} |")
            else:
                report_lines.append("⚠️ 无算子数据，请检查 Profiling 数据完整性")
        except Exception as e:
            report_lines.append(f"⚠️ 获取算子数据失败: {e}")
        report_lines.append("")

    except Exception as e:
        report_lines.append(f"⚠️ 加载 Profiling 数据时出现错误: {e}")
        import traceback
        report_lines.append(f"```\n{traceback.format_exc()}\n```")
    report_lines.append("")

    # 4. Timeline 详细分析
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 4. Timeline 详细分析")
    report_lines.append("")

    if trace_view_path:
        try:
            print("正在进行 Timeline 摘要分析...")
            parser = StreamParser(str(trace_view_path))
            summarizer = TimelineSummarizer(max_top_events=50)

            event_count = 0
            for event in parser.iter_events(show_progress=True):
                summarizer.process_event(event)
                event_count += 1

            summary = summarizer.get_summary()

            report_lines.append("### 事件统计")
            report_lines.append("")
            report_lines.append(f"- **总事件数**: {summary['total_events']}")
            report_lines.append(f"- **总持续时间**: {summary['total_duration_us'] / 1000:.2f} ms")

            # 按 Category 分组
            if summary['by_category']:
                report_lines.append("")
                report_lines.append("### 按分类统计")
                report_lines.append("")
                report_lines.append("| Category | 事件数 | 总耗时 (ms) | 平均耗时 (μs) |")
                report_lines.append("|----------|--------|-------------|---------------|")
                for cat, stats in sorted(summary['by_category'].items(),
                                         key=lambda x: x[1]['duration'], reverse=True):
                    dur_ms = stats['duration'] / 1000
                    avg_us = stats['duration'] / stats['count'] if stats['count'] > 0 else 0
                    report_lines.append(f"| {cat} | {stats['count']} | {dur_ms:.2f} | {avg_us:.2f} |")

            # Top 事件
            if summary['top_by_duration']:
                report_lines.append("")
                report_lines.append("### Top 耗时事件 (Top 30)")
                report_lines.append("")
                report_lines.append("| 排名 | 事件名称 | Category | 耗时 (ms) |")
                report_lines.append("|------|----------|----------|----------|")
                for i, (dur, name, cat) in enumerate(sorted(summary['top_by_duration'],
                                                             reverse=True)[:30], 1):
                    dur_ms = dur / 1000
                    name_short = name[:40] + "..." if len(name) > 40 else name
                    report_lines.append(f"| {i} | {name_short} | {cat} | {dur_ms:.2f} |")

            # 质量报告
            quality = parser.get_quality_report()
            if quality:
                report_lines.append("")
                report_lines.append("### 数据质量")
                report_lines.append("")
                report_lines.append(f"- **有效事件**: {quality.valid_events}")
                report_lines.append(f"- **跳过事件**: {quality.skipped_events}")
                report_lines.append(f"- **错误数**: {quality.error_count}")
                report_lines.append(f"- **警告数**: {quality.warning_count}")

        except Exception as e:
            report_lines.append(f"⚠️ Timeline 分析失败: {e}")
            import traceback
            report_lines.append(f"```\n{traceback.format_exc()}\n```")
    else:
        report_lines.append("⚠️ 未找到 trace_view.json，跳过 Timeline 分析")
    report_lines.append("")

    # 5. 性能瓶颈分析
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 5. 性能瓶颈分析")
    report_lines.append("")

    # 基于 step trace 分析瓶颈
    if 'step_trace' in locals() and step_trace is not None and not step_trace.empty:
        report_lines.append("### Step 级别瓶颈")
        report_lines.append("")

        # 计算各阶段占比
        if 'free' in step_trace.columns:
            free_time = step_trace['free'].mean()
            total = step_trace[available_time_cols].sum(axis=1).mean() if available_time_cols else 1
            free_pct = (free_time / total * 100) if total > 0 else 0

            if free_pct > 30:
                report_lines.append(f"🔴 **严重**: 空闲时间占比过高 ({free_pct:.1f}%)，可能存在数据加载瓶颈")
            elif free_pct > 15:
                report_lines.append(f"🟡 **警告**: 空闲时间较高 ({free_pct:.1f}%)，建议优化数据加载")
            else:
                report_lines.append(f"🟢 **良好**: 空闲时间占比 ({free_pct:.1f}%)")

        if 'communication_not_overlapped' in step_trace.columns:
            comm_time = step_trace['communication_not_overlapped'].mean()
            total = step_trace[available_time_cols].sum(axis=1).mean() if available_time_cols else 1
            comm_pct = (comm_time / total * 100) if total > 0 else 0

            if comm_pct > 20:
                report_lines.append(f"🔴 **严重**: 未掩盖通信时间过高 ({comm_pct:.1f}%)，建议优化通信策略")
            elif comm_pct > 10:
                report_lines.append(f"🟡 **警告**: 存在未掩盖通信 ({comm_pct:.1f}%)")

        # 计算 MFU
        if 'computing' in step_trace.columns:
            computing_time = step_trace['computing'].mean()
            if hw_spec and computing_time > 0:
                # 简单估算：假设实际计算时间占比
                compute_ratio = computing_time / total if total > 0 else 0
                estimated_mfu = compute_ratio * 0.8  # 假设80%的计算效率
                report_lines.append("")
                report_lines.append(f"### MFU 估算")
                report_lines.append("")
                report_lines.append(f"- **估算 MFU**: {estimated_mfu * 100:.1f}%")
                report_lines.append(f"- **计算占比**: {compute_ratio * 100:.1f}%")
                report_lines.append("")
                if estimated_mfu < 0.3:
                    report_lines.append("🔴 **MFU 较低**，建议检查：")
                    report_lines.append("  - 算子是否已充分利用 NPU 特性")
                    report_lines.append("  - 是否存在 CPU fallback")
                    report_lines.append("  - 内存带宽是否成为瓶颈")
                elif estimated_mfu < 0.5:
                    report_lines.append("🟡 **MFU 中等**，有优化空间")
                else:
                    report_lines.append("🟢 **MFU 良好**")

    report_lines.append("")

    # 6. 优化建议
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 6. 优化建议")
    report_lines.append("")
    report_lines.append("基于以上分析，推荐的优化方向：")
    report_lines.append("")

    # 根据分析结果生成具体建议
    suggestions = []

    if 'step_trace' in locals() and step_trace is not None and not step_trace.empty:
        if 'free' in step_trace.columns:
            free_pct = (step_trace['free'].mean() / step_trace[available_time_cols].sum(axis=1).mean() * 100) if available_time_cols else 0
            if free_pct > 20:
                suggestions.append("1. **数据加载优化**")
                suggestions.append(f"   - 空闲时间占比 {free_pct:.1f}%，可能存在数据加载瓶颈")
                suggestions.append("   - 增加 DataLoader workers 数量")
                suggestions.append("   - 启用数据预取 (prefetch)")
                suggestions.append("   - 检查 I/O 是否成为瓶颈")

        if 'communication_not_overlapped' in step_trace.columns:
            comm_pct = (step_trace['communication_not_overlapped'].mean() / step_trace[available_time_cols].sum(axis=1).mean() * 100) if available_time_cols else 0
            if comm_pct > 5:
                suggestions.append("2. **通信优化**")
                suggestions.append(f"   - 未掩盖通信时间占比 {comm_pct:.1f}%")
                suggestions.append("   - 启用梯度累积以减少通信频率")
                suggestions.append("   - 使用通信算子融合")
                suggestions.append("   - 调整重计算策略以增加计算通信比")

    if 'top_kernels' in locals() and top_kernels:
        suggestions.append("3. **算子优化**")
        suggestions.append(f"   - Top 1 算子 `{top_kernels[0].get('name', 'N/A')}` 耗时 {top_kernels[0].get('total_dur_us', 0) / 1000:.2f} ms")
        suggestions.append("   - 检查是否可以使用 NPU 原生算子替代")
        suggestions.append("   - 考虑算子融合以减少 kernel launch 开销")
        suggestions.append("   - 检查是否存在不必要的 CPU-GPU 数据传输")

    if not suggestions:
        suggestions = [
            "1. **Kernel 优化**：优化耗时 Top 算子，查看是否可以融合或替换",
            "2. **通信优化**：调整通信策略，提高计算与通信的掩盖率",
            "3. **内存优化**：关注内存碎片化和可能的内存泄漏",
            "4. **并行策略**：检查并行策略是否合理，避免负载不均",
        ]

    for suggestion in suggestions:
        report_lines.append(suggestion)
    report_lines.append("")

    # 7. 附录
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 7. 附录")
    report_lines.append("")
    report_lines.append("### 7.1 数据来源")
    report_lines.append("")
    report_lines.append(f"- **Profiling 工具**: msprof")
    report_lines.append(f"- **分析工具**: npu-mfu-analyzer")
    report_lines.append(f"- **硬件**: {hw_spec.name if hw_spec else 'Unknown'}")
    report_lines.append("")

    report_lines.append("### 7.2 术语解释")
    report_lines.append("")
    report_lines.append("| 术语 | 说明 |")
    report_lines.append("|------|------|")
    report_lines.append("| MFU | Model FLOPS Utilization，模型算力利用率 |")
    report_lines.append("| AICore | 昇腾 NPU 的计算核心 |")
    report_lines.append("| HBM | High Bandwidth Memory，高带宽内存 |")
    report_lines.append("| HCCS | 昇腾芯片间高速互联 |")
    report_lines.append("| Overlap | 通信掩盖，通信与计算重叠的比例 |")
    report_lines.append("| Free | 空闲时间，可能用于数据加载或调度 |")
    report_lines.append("")

    # 写入报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n报告已生成: {output_file}")
    print(f"报告大小: {output_file.stat().st_size / 1024:.1f} KB")

    return "\n".join(report_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python scripts/detailed_analysis.py <profiling_path> [output_path]")
        sys.exit(1)

    profiling_path = sys.argv[1]

    # 默认输出到上级目录
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = str(Path(__file__).parent.parent.parent / f"npu_detailed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    analyze_profiling(profiling_path, output_path)
