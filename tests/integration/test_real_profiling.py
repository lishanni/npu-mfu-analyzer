#!/usr/bin/env python3
"""
使用真实 Profiling 数据验证融合算子分析和 AIKG 流程

测试 hyper-gitcode 中的真实 profiling 数据
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from npu_mfu_analyzer.agents.operator_agent import FusionAnalyzer, AIKGIntegrator, AIKGRequestConverter
from npu_mfu_analyzer.agents.fusion_rules import FusionOpportunity
from npu_mfu_analyzer.data_loader.profiling_loader import ProfilingLoader


def test_real_profiling_fusion_analysis():
    """测试真实 Profiling 数据的融合分析"""
    profiling_path = "/Users/dengyiping/Desktop/hyper-gitcode/msprof_3813_20260124100335956_ascend_pt-1"

    print(f"\n{'='*60}")
    print(f"测试真实 Profiling 数据: {profiling_path}")
    print(f"{'='*60}\n")

    # 1. 解析 Profiling 数据
    print("[1/4] 解析 Profiling 数据...")
    loader = ProfilingLoader(profiling_path)

    # 获取算子信息（获取更多算子用于融合分析）
    all_operators = loader.get_top_kernels(top_n=100)

    if not all_operators:
        print("  ❌ 未找到算子数据")
        return False

    print(f"  ✅ 成功解析 {len(all_operators)} 个算子")

    # 显示前 10 个算子
    print("\n  Top 10 耗时算子:")
    for i, op in enumerate(all_operators[:10], 1):
        name = op.get("name", "unknown")[:40]
        dur_ms = op.get("dur", 0) / 1000
        print(f"    {i}. {name}: {dur_ms:.2f} ms")

    # 2. 融合机会分析
    print("\n[2/4] 检测融合机会...")
    analyzer = FusionAnalyzer()

    # 检测融合机会（all_operators 已经是列表格式）
    fusion_opportunities = analyzer.detect_opportunities(
        all_operators=all_operators,
        timeline_data=None
    )

    print(f"  ✅ 检测到 {len(fusion_opportunities)} 个融合机会")

    if not fusion_opportunities:
        print("  ⚠️  未检测到融合机会")
        return True

    # 显示融合机会
    print("\n  融合机会详情:")
    for i, opp in enumerate(fusion_opportunities[:5], 1):
        print(f"\n    [{i}] {opp.name}")
        print(f"        类型: {opp.opportunity_type}")
        print(f"        算子级别加速: {opp.estimated_speedup:.1f}x")
        print(f"        端到端加速: {opp.end_to_end_speedup:.1%}")
        print(f"        时间占比: {opp.time_proportion:.1%}")
        print(f"        复杂度: {opp.complexity}")
        if opp.ascend_op:
            print(f"        昇腾算子: {opp.ascend_op}")
        print(f"        涉及算子数: {len(opp.current_ops)}")

    # 3. AIKG 请求转换
    print("\n[3/4] AIKG 请求转换...")
    # 临时使用低阈值进行验证
    converter = AIKGRequestConverter(
        min_speedup_threshold=1.0,  # 临时设置最低阈值，不过滤
        max_complexity="高",
        skip_native_ops=False  # 临时不过滤昇腾已有算子，验证完整流程
    )

    aikg_requests = converter.convert_opportunities(fusion_opportunities)

    print(f"  ✅ 转换生成 {len(aikg_requests)} 个 AIKG 请求")

    if aikg_requests:
        print("\n  AIKG 请求列表:")
        for i, req in enumerate(aikg_requests[:5], 1):
            print(f"\n    [{i}] {req.fusion_name}")
            print(f"        算子序列: {' -> '.join(req.operator_sequence[:3])}")
            print(f"        目标加速: {req.target_speedup:.1f}x")
            print(f"        复杂度: {req.complexity}")
    else:
        print("  ℹ️  所有融合机会被过滤（昇腾已有算子或加速比过低）")

    # 4. 生成 AIKG Prompt 示例
    if aikg_requests:
        print("\n[4/4] AIKG Prompt 示例...")
        print("\n  第一个请求的 Prompt:")
        print("-" * 60)
        prompt = aikg_requests[0].to_aikg_prompt()
        # 只显示前 1000 个字符
        print(prompt[:1000])
        if len(prompt) > 1000:
            print(f"\n  ... (省略 {len(prompt) - 1000} 个字符)")
        print("-" * 60)

        # 5. AIKG 客户端测试（可选，不实际调用 LLM）
        print("\n[5/5] AIKG 客户端测试...")
        from npu_mfu_analyzer.agents.aikg_integration import AIKGKernelClient

        # 创建客户端（不配置 LLM，用于测试流程）
        client = AIKGKernelClient(
            service_url=None,  # 不使用远程服务
            llm_client=None,   # 不使用 LLM
        )

        # 测试代码块提取功能
        test_llm_response = '''
        当然，这是您需要的融合算子 Triton 代码：

        ```python
        import torch
        import triton
        from triton import language as tl

        @triton.jit
        def fused_matmul_bias_gelu(
            x_ptr, y_ptr, bias_ptr, output_ptr,
            M, N, K,
            stride_xm, stride_xk,
            stride_yk, stride_yn,
            stride_bias,
            stride_om, stride_on,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
        ):
            # Fused MatMul + Bias + GELU implementation
            pid = tl.program_id(axis=0)
            # ... (省略实现细节)

        def fused_matmul_bias_gelu_call(x, y, bias):
            return fused_matmul_bias_gelu[grid](x, y, bias)
        ```

        以上代码实现了 MatMul、Bias Add 和 GELU 激活的融合。
        '''

        extracted_code = client._extract_code_block(test_llm_response, "python")
        if extracted_code:
            print("  ✅ 代码块提取功能正常")
            print(f"\n  提取的代码预览:")
            lines = extracted_code.split("\n")[:10]
            for line in lines:
                print(f"    {line}")
            if len(extracted_code.split("\n")) > 10:
                print(f"    ... (省略 {len(extracted_code.split('\n')) - 10} 行)")
        else:
            print("  ❌ 代码块提取失败")

        # 测试编译脚本生成
        build_script = client._generate_build_script(aikg_requests[0])
        if build_script and "#!/bin/bash" in build_script:
            print("\n  ✅ 编译脚本生成功能正常")
        else:
            print("\n  ❌ 编译脚本生成失败")

        # 测试性能测试代码生成
        benchmark_code = client._generate_benchmark_code(aikg_requests[0])
        if benchmark_code and "benchmark" in benchmark_code.lower():
            print("  ✅ 性能测试代码生成功能正常")
        else:
            print("  ❌ 性能测试代码生成失败")

        # 6. 保存到文件（临时目录）
        print("\n[6/6] 文件保存测试...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from npu_mfu_analyzer.agents.aikg_integration import GeneratedKernel, GenerationStatus

            # 模拟一个生成的内核
            mock_kernel = GeneratedKernel(
                kernel_name=aikg_requests[0].fusion_name,
                request_id=aikg_requests[0].original_opportunity_id,
                triton_code=extracted_code or "# Mock Triton code",
                build_script=build_script,
                benchmark_code=benchmark_code,
                status=GenerationStatus.SUCCESS,
                estimated_speedup=aikg_requests[0].target_speedup,
            )

            # 模拟保存
            kernel_dir = Path(tmpdir) / "kernels"
            kernel_dir.mkdir(parents=True, exist_ok=True)

            safe_name = aikg_requests[0].fusion_name.replace(" ", "_").replace("-", "_").lower()
            base_path = kernel_dir / safe_name

            # 保存 Triton 代码
            if mock_kernel.triton_code:
                triton_path = base_path.with_suffix(".py")
                triton_path.write_text(mock_kernel.triton_code, encoding="utf-8")
                print(f"  ✅ Triton 代码已保存: {triton_path}")
                print(f"      文件大小: {triton_path.stat().st_size} bytes")

            # 保存编译脚本
            if mock_kernel.build_script:
                build_path = base_path.with_suffix(".sh")
                build_path.write_text(mock_kernel.build_script, encoding="utf-8")
                build_path.chmod(0o755)
                print(f"  ✅ 编译脚本已保存: {build_path}")
                print(f"      文件大小: {build_path.stat().st_size} bytes")

            # 保存性能测试代码
            if mock_kernel.benchmark_code:
                bench_path = kernel_dir / f"{safe_name}_bench.py"
                bench_path.write_text(mock_kernel.benchmark_code, encoding="utf-8")
                bench_path.chmod(0o755)
                print(f"  ✅ 性能测试代码已保存: {bench_path}")
                print(f"      文件大小: {bench_path.stat().st_size} bytes")

    return True


async def test_aikg_generation_flow():
    """测试 AIKG 生成流程（不实际调用 LLM）"""
    print(f"\n{'='*60}")
    print("测试 AIKG 生成流程（模拟）")
    print(f"{'='*60}\n")

    # 创建一些模拟的融合机会
    opportunities = [
        FusionOpportunity(
            opportunity_type="fuse",
            name="MatMul+Bias+GELU 融合",
            description="将矩阵乘法、偏置加法和 GELU 激活融合",
            current_ops=[
                {"name": "MatMulV2_L1_QK", "dur": 250000},
                {"name": "BiasAdd_L1_QK", "dur": 15000},
                {"name": "GELU_L1", "dur": 20000},
            ],
            estimated_speedup=1.3,
            end_to_end_speedup=1.08,
            time_proportion=0.12,
            memory_saving=0.4,
            implementation="Triton 实现",
            complexity="中等",
        ),
        FusionOpportunity(
            opportunity_type="fuse",
            name="LayerNorm+Residual Add 融合",
            description="将归一化和残差连接融合",
            current_ops=[
                {"name": "RMSNorm_L1", "dur": 30000},
                {"name": "Add_residual_L1", "dur": 10000},
            ],
            estimated_speedup=1.2,
            end_to_end_speedup=1.03,
            time_proportion=0.05,
            memory_saving=0.3,
            implementation="使用昇腾 aclnnAddRmsNorm",
            complexity="低",
            ascend_op="aclnnAddRmsNorm",
        ),
    ]

    print("[1/3] 创建 AIKG 集成器...")

    converter = AIKGRequestConverter(
        min_speedup_threshold=1.05,
        skip_native_ops=False  # 不跳过，用于演示
    )

    # 注意：这里不创建 client，因为我们不想实际调用 LLM
    print("  ✅ AIKG 请求转换器创建成功")

    # 转换为 AIKG 请求
    print("\n[2/3] 转换融合机会...")

    aikg_requests = converter.convert_opportunities(opportunities)

    print(f"  ✅ 转换生成 {len(aikg_requests)} 个 AIKG 请求")

    for i, req in enumerate(aikg_requests, 1):
        print(f"\n  请求 {i}: {req.fusion_name}")
        print(f"    - 目标加速: {req.target_speedup:.1f}x")
        print(f"    - 算子数: {len(req.operator_sequence)}")
        print(f"    - 复杂度: {req.complexity}")

    # 生成 Prompt
    print("\n[3/3] 生成 AIKG Prompt...")

    for i, req in enumerate(aikg_requests, 1):
        prompt = req.to_aikg_prompt()
        print(f"\n  请求 {i} 的 Prompt 预览:")
        print("  " + "-" * 56)
        lines = prompt.split("\n")[:15]  # 只显示前 15 行
        for line in lines:
            print(f"  {line}")
        if len(prompt.split("\n")) > 15:
            print(f"  ... (省略其余 {len(prompt.split('\n')) - 15} 行)")
        print("  " + "-" * 56)

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NPU MFU Analyzer - 真实 Profiling 数据验证测试")
    print("=" * 60)

    # 测试 1: 真实 Profiling 数据分析
    try:
        success = test_real_profiling_fusion_analysis()
        if not success:
            print("\n❌ 真实 Profiling 数据分析测试失败")
            return 1
    except Exception as e:
        print(f"\n❌ 真实 Profiling 数据分析出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 测试 2: AIKG 生成流程
    try:
        success = asyncio.run(test_aikg_generation_flow())
        if not success:
            print("\n❌ AIKG 生成流程测试失败")
            return 1
    except Exception as e:
        print(f"\n❌ AIKG 生成流程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
