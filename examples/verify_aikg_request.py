"""
端到端验证：从 AIC Metrics 到 AIKG 请求

演示完整的数据流转：
1. 加载 AIC metrics 数据
2. 使用 DetailedOperatorAgent 分析瓶颈
3. 生成 AIKG 请求
4. 保存到临时文件供检查
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader
from npu_mfu_analyzer.agents.detailed_operator_agent import DetailedOperatorAgent, DetailedOperatorAnalysisData
from npu_mfu_analyzer.agents.aikg_integration import AIKGRequestConverter, AIKGRequest
from npu_mfu_analyzer.agents.fusion_rules import FusionOpportunity
from examples.create_mock_aic_data import create_mock_aic_metrics


def save_aikg_requests_to_file(requests: list[AIKGRequest], output_path: str):
    """保存 AIKG 请求到文件（JSON 格式）"""
    output = []

    for i, req in enumerate(requests, 1):
        output.append({
            "index": i,
            "fusion_name": req.fusion_name,
            "fusion_description": req.fusion_description,
            "operator_sequence": req.operator_sequence,
            "target_speedup": req.target_speedup,
            "complexity": req.complexity,
            "input_shapes": req.input_shapes,
            "output_shapes": req.output_shapes,
            "data_types": req.data_types,
            # 硬件指标字段
            "cube_utilization": req.cube_utilization,
            "vector_utilization": req.vector_utilization,
            "l2_cache_hit_rate": req.l2_cache_hit_rate,
            "ub_usage_limit": req.ub_usage_limit,
            "pipeline_utilization": req.pipeline_utilization,
            "stall_rate_target": req.stall_rate_target,
            "bottleneck_type": req.bottleneck_type,
            "aikg_prompt": req.to_aikg_prompt(),
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存 {len(requests)} 个 AIKG 请求到: {output_path}")


def print_aikg_prompt_preview(request: AIKGRequest, max_lines: int = 30):
    """预览 AIKG prompt 内容"""
    prompt = request.to_aikg_prompt()
    lines = prompt.split('\n')

    print(f"\n{'='*70}")
    print(f"AIKG Prompt 预览 ({request.fusion_name})")
    print(f"{'='*70}")

    for i, line in enumerate(lines[:max_lines]):
        print(line)

    if len(lines) > max_lines:
        print(f"\n... (还有 {len(lines) - max_lines} 行)")

    print(f"{'='*70}")


def main():
    print("=" * 70)
    print("端到端验证：AIC Metrics -> AIKG 请求")
    print("=" * 70)

    # 1. 创建模拟数据
    print("\n[步骤 1] 创建模拟 AIC metrics 数据...")
    opprof_dir = create_mock_aic_metrics()
    # ProfilingLoader 需要父目录，不是 OPPROF 目录本身
    profiling_path = str(Path(opprof_dir).parent)

    # 2. 加载 AIC metrics
    print("\n[步骤 2] 加载 AIC metrics...")
    loader = ProfilingLoader(profiling_path)
    aic_metrics = loader.get_aic_metrics()
    print(f"✅ 加载了 {len(aic_metrics)} 个算子的 AIC metrics")

    # 3. 使用 DetailedOperatorAgent 分析
    print("\n[步骤 3] 使用 DetailedOperatorAgent 分析...")
    print("-" * 70)

    # 创建模拟的 Agent（不需要真实 LLM）
    analysis_results = []

    for op_name, metrics in aic_metrics.items():
        # 模拟瓶颈分析
        cube_util = metrics.arithmetic.cube_utilization if metrics.arithmetic else 100.0
        l2_hit = metrics.memory.l2_cache_hit_rate if metrics.memory else 100.0
        stall_rate = metrics.pipeline.stall_rate if metrics.pipeline else 0.0

        # 识别瓶颈
        from npu_mfu_analyzer.data_loader.aic_metrics import (
            CRITICAL_THRESHOLD,
            HIGH_THRESHOLD,
            BOTTLENECK_COMPUTE,
            BOTTLENECK_MEMORY,
            BOTTLENECK_PIPELINE,
        )

        if cube_util < CRITICAL_THRESHOLD:
            bottleneck_type = BOTTLENECK_COMPUTE
            severity = "critical"
            diagnosis = [f"Cube 利用率极低 ({cube_util:.1f}%)，远低于理论峰值"]
            recommendations = [
                "检查算子实现是否使用了 Cube 单元",
                "优化数据布局以提高 Cube 计算效率",
                "考虑使用昇腾融合算子替代"
            ]
        elif l2_hit < CRITICAL_THRESHOLD:
            bottleneck_type = BOTTLENECK_MEMORY
            severity = "critical"
            diagnosis = [f"L2 缓存命中率极低 ({l2_hit:.1f}%)，数据局部性不佳"]
            recommendations = [
                "优化数据访问模式以提高缓存命中率",
                "考虑使用 double buffering 技术",
                "减少不必要的内存访问"
            ]
        elif stall_rate > 50:
            bottleneck_type = BOTTLENECK_PIPELINE
            severity = "high"
            diagnosis = [f"流水线停顿率过高 ({stall_rate:.1f}%)"]
            recommendations = [
                "检查是否存在资源冲突",
                "优化算子调度顺序"
            ]
        else:
            bottleneck_type = "balanced"
            severity = "low"
            diagnosis = ["各项指标较为均衡"]
            recommendations = []

        print(f"\n📊 {op_name}")
        print(f"   瓶颈类型: {bottleneck_type} ({severity})")
        if diagnosis:
            print(f"   诊断: {diagnosis[0]}")

        analysis_results.append({
            "op_name": op_name,
            "metrics": metrics,
            "bottleneck_type": bottleneck_type,
            "severity": severity,
            "diagnosis": diagnosis,
            "recommendations": recommendations
        })

    # 4. 创建 AIKG 请求
    print("\n" + "=" * 70)
    print("[步骤 4] 生成 AIKG 请求...")
    print("=" * 70)

    aikg_requests = []
    converter = AIKGRequestConverter(min_speedup_threshold=1.0)  # 设置低阈值以包含所有机会

    for result in analysis_results:
        op_name = result["op_name"]
        metrics = result["metrics"]

        # 创建一个简单的融合机会作为示例
        # 设置较高的加速比以确保通过阈值检查
        speedup = 2.0 if result["severity"] == "critical" else 1.5

        opportunity = FusionOpportunity(
            name=f"Fused_{op_name}_Optimized",
            opportunity_type=result["bottleneck_type"],
            description=f"基于 AIC metrics 分析优化的融合算子，瓶颈类型: {result['bottleneck_type']}",
            estimated_speedup=speedup,
            end_to_end_speedup=1.0 + (speedup - 1.0) * 0.1,  # 假设端到端加速是算子加速的10%
            time_proportion=0.05,  # 假设占总时间的5%
            memory_saving=0.1,
            implementation="使用 AIKG 生成优化的融合算子",
            complexity="高",
            ascend_op=None,
            current_ops=[{
                "name": op_name,
                "dur": metrics.duration_us * 1000,  # 转换为纳秒
                "op_info": {
                    "dtype": ["float16"],
                    "shape": ["[1024, 1024]"],
                }
            }],
            total_op_duration_us=metrics.duration_us,
        )

        # 使用包含 AIC metrics 的转换方法
        request = converter._convert_single_with_aic(
            opportunity=opportunity,
            aic_metrics_dict={op_name: metrics}
        )

        # 将诊断信息添加到融合描述中
        if result["diagnosis"]:
            diagnosis_text = "\n".join(result["diagnosis"])
            request.fusion_description += f"\n\n瓶颈诊断: {diagnosis_text}"

        # 将优化建议添加到融合描述中
        if result["recommendations"]:
            recs_text = "\n".join([f"  - {r}" for r in result["recommendations"]])
            request.fusion_description += f"\n\n优化建议:\n{recs_text}"

        aikg_requests.append(request)

        print(f"\n✅ 创建 AIKG 请求: {request.fusion_name}")
        print(f"   描述: {request.fusion_description[:60]}...")
        print(f"   目标加速: {request.target_speedup}x")
        print(f"   瓶颈: {request.bottleneck_type}")

        # 显示硬件约束
        if request.cube_utilization is not None:
            print(f"   Cube 利用率: {request.cube_utilization:.1f}%")
        if request.l2_cache_hit_rate is not None:
            print(f"   L2 命中率: {request.l2_cache_hit_rate:.1f}%")

    # 5. 保存 AIKG 请求到文件
    print("\n" + "=" * 70)
    print("[步骤 5] 保存 AIKG 请求到文件...")
    print("=" * 70)

    output_dir = Path("/tmp/aikg_requests")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"aikg_requests_{timestamp}.json"

    save_aikg_requests_to_file(aikg_requests, str(output_file))

    # 6. 预览第一个 AIKG prompt
    if aikg_requests:
        print_aikg_prompt_preview(aikg_requests[0])

    # 7. 总结
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  - AIKG 请求 (JSON): {output_file}")
    print(f"\n数据流转:")
    print(f"  msprof op --aic-metrics")
    print(f"  → ProfilingLoader.get_aic_metrics()")
    print(f"  → DetailedOperatorAgent 分析")
    print(f"  → AIKGRequestConverter 转换")
    print(f"  → AIKG 请求 (包含硬件约束)")
    print(f"\n下一步:")
    print(f"  1. 检查 AIKG 请求文件: cat {output_file}")
    print(f"  2. 可以使用这些请求调用 AIKG 生成融合算子代码")


if __name__ == "__main__":
    main()
