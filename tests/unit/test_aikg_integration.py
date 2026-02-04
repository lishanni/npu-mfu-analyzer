"""
AIKG 集成模块单元测试

测试 AIKG 请求转换、算子生成和文件保存功能。
"""

import pytest
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict

from src.agents.aikg_integration import (
    AIKGRequest,
    GeneratedKernel,
    GenerationStatus,
    AIKGBackend,
    AIKGRequestConverter,
    AIKGKernelClient,
    AIKGIntegrator,
)
from src.agents.fusion_rules import FusionOpportunity


class TestAIKGRequest:
    """AIKGRequest 测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        request = AIKGRequest(
            fusion_name="Test Fusion",
            fusion_description="Test description",
            operator_sequence=["MatMul", "Add"],
            operator_pattern="MatMul -> Add",
            input_shapes=[[1024, 1024]],
            output_shapes=[[1024, 1024]],
            data_types=["FP16"],
            target_speedup=1.5,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id="test123",
            complexity="低",
            estimated_memory_saving=0.3,
        )

        d = request.to_dict()

        assert d["fusion_name"] == "Test Fusion"
        assert d["target_backend"] == "ascend"
        assert d["target_speedup"] == 1.5
        assert d["complexity"] == "低"

    def test_to_aikg_prompt(self):
        """测试生成 AIKG Prompt"""
        request = AIKGRequest(
            fusion_name="MatMul+Add Fusion",
            fusion_description="Fuse MatMul and Add operations",
            operator_sequence=["MatMul", "BiasAdd"],
            operator_pattern="MatMul -> BiasAdd",
            input_shapes=[[1024, 1024], [1024]],
            output_shapes=[[1024, 1024]],
            data_types=["FP16"],
            target_speedup=1.3,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id="test123",
            complexity="中等",
            estimated_memory_saving=0.4,
        )

        prompt = request.to_aikg_prompt()

        # 验证关键内容
        assert "MatMul+Add Fusion" in prompt
        assert "Fusion Name" in prompt
        assert "MatMul" in prompt
        assert "BiasAdd" in prompt
        assert "FP16" in prompt
        assert "ascend" in prompt
        assert "1.3x" in prompt
        assert "40%" in prompt  # 内存节省
        assert "Triton" in prompt


class TestAIKGRequestConverter:
    """AIKGRequestConverter 测试"""

    def setup_method(self):
        """测试前设置"""
        self.converter = AIKGRequestConverter(
            min_speedup_threshold=1.05,
            max_complexity="高",
            skip_native_ops=True
        )

    def test_convert_matmul_bias_fusion(self):
        """测试转换 MatMul+Bias 融合机会"""
        opportunity = FusionOpportunity(
            opportunity_type="fuse",
            name="MatMul+Bias+Activation 融合",
            description="融合矩阵乘法和偏置加法",
            current_ops=[
                {"name": "MatMulV2_1", "dur": 200000, "input_shapes": "[1024,1024];[1024,1024]"},
                {"name": "BiasAdd_1", "dur": 10000, "input_shapes": "[1024,1024]"},
                {"name": "GELU_1", "dur": 15000},
            ],
            estimated_speedup=1.3,
            end_to_end_speedup=1.25,  # 高于阈值 1.2
            time_proportion=0.15,
            memory_saving=0.4,
            implementation="使用昇腾 aclnnFusedMatMulBiasAct",
            complexity="低",
            ascend_op=None,  # 不跳过
        )

        requests = self.converter.convert_opportunities([opportunity])

        assert len(requests) == 1
        request = requests[0]

        assert request.fusion_name == "MatMul+Bias+Activation 融合"
        assert "MatMul" in request.operator_sequence[0]
        assert request.target_speedup == 1.3
        assert request.complexity == "低"
        assert request.estimated_memory_saving == 0.4

    def test_skip_low_speedup_opportunity(self):
        """测试跳过低加速比的融合机会"""
        opportunity = FusionOpportunity(
            opportunity_type="fuse",
            name="Low Speedup Fusion",
            description="加速比太低",
            current_ops=[{"name": "Op1", "dur": 1000}],
            estimated_speedup=1.0,  # 低于阈值
            end_to_end_speedup=1.01,  # 低于阈值
            time_proportion=0.01,
            memory_saving=0.1,
            implementation="Test",
            complexity="低",
        )

        requests = self.converter.convert_opportunities([opportunity])

        assert len(requests) == 0  # 应该被跳过

    def test_skip_native_ascend_ops(self):
        """测试跳过昇腾已有算子"""
        opportunity = FusionOpportunity(
            opportunity_type="replace",
            name="FlashAttention 替换",
            description="昇腾已有",
            current_ops=[{"name": "Attention", "dur": 100000}],
            estimated_speedup=5.0,
            end_to_end_speedup=1.2,
            time_proportion=0.2,
            memory_saving=0.6,
            implementation="昇腾原生",
            complexity="低",
            ascend_op="aclnnFlashAttentionScore",  # 有昇腾算子
        )

        # 设置 skip_native_ops=True
        converter = AIKGRequestConverter(skip_native_ops=True)
        requests = converter.convert_opportunities([opportunity])

        assert len(requests) == 0  # 应该被跳过

        # 设置 skip_native_ops=False
        converter = AIKGRequestConverter(skip_native_ops=False)
        requests = converter.convert_opportunities([opportunity])

        assert len(requests) == 1  # 不跳过

    def test_skip_high_complexity(self):
        """测试跳过高复杂度"""
        opportunity = FusionOpportunity(
            opportunity_type="custom",
            name="Complex Fusion",
            description="太复杂",
            current_ops=[{"name": "Op1", "dur": 1000}],
            estimated_speedup=2.0,
            end_to_end_speedup=1.1,
            time_proportion=0.1,
            memory_saving=0.5,
            implementation="复杂",
            complexity="高",  # 高复杂度
        )

        # 设置 max_complexity="中等"
        converter = AIKGRequestConverter(max_complexity="中等")
        requests = converter.convert_opportunities([opportunity])

        assert len(requests) == 0  # 应该被跳过

    def test_convert_multiple_opportunities(self):
        """测试转换多个融合机会"""
        opportunities = [
            FusionOpportunity(
                opportunity_type="fuse",
                name=f"Fusion {i}",
                description=f"Description {i}",
                current_ops=[{"name": f"Op{i}", "dur": 10000 * (i + 1)}],
                estimated_speedup=1.3 + i * 0.1,
                end_to_end_speedup=1.25 + i * 0.05,  # 高于阈值 1.2
                time_proportion=0.1,
                memory_saving=0.3,
                implementation="Test",
                complexity="低",
            )
            for i in range(5)
        ]

        requests = self.converter.convert_opportunities(opportunities)

        assert len(requests) == 5
        for i, req in enumerate(requests):
            assert req.fusion_name == f"Fusion {i}"
            assert req.target_speedup == 1.3 + i * 0.1


class TestAIKGKernelClient:
    """AIKGKernelClient 测试"""

    def setup_method(self):
        """测试前设置"""
        self.client = AIKGKernelClient()

    def test_init(self):
        """测试初始化"""
        client = AIKGKernelClient(
            service_url="http://localhost:8080",
            timeout=600,
            max_concurrent=5
        )

        assert client.service_url == "http://localhost:8080"
        assert client.timeout == 600
        assert client.max_concurrent == 5

    def test_stats(self):
        """测试统计信息"""
        stats = self.client.get_stats()

        assert "total_requests" in stats
        assert "success_count" in stats
        assert "failed_count" in stats
        assert "skipped_count" in stats

    @pytest.mark.asyncio
    async def test_generate_without_config(self):
        """测试无配置时的生成"""
        request = AIKGRequest(
            fusion_name="Test",
            fusion_description="Test",
            operator_sequence=["Op1"],
            operator_pattern="Op1",
            input_shapes=[[100, 100]],
            output_shapes=[[100, 100]],
            data_types=["FP16"],
            target_speedup=1.5,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id="test",
            complexity="低",
            estimated_memory_saving=0.3,
        )

        kernels = await self.client.generate_kernels([request])

        assert len(kernels) == 1
        assert kernels[0].status == GenerationStatus.SKIPPED
        assert kernels[0].error_message == "No AIKG service or LLM configured"

    def test_extract_code_block(self):
        """测试代码块提取"""
        text = '''
        Some text before

        ```python
        import triton
        import torch

        @triton.jit
        def kernel(x, y):
            pass
        ```

        Some text after
        '''

        code = self.client._extract_code_block(text, "python")

        assert code is not None
        assert "import triton" in code
        assert "@triton.jit" in code
        assert "```" not in code

    def test_extract_code_block_no_delimiters(self):
        """测试没有分隔符的代码提取"""
        text = '''
        def kernel(x, y):
            return x + y
        '''

        code = self.client._extract_code_block(text, "python")

        # 应该返回原始文本（因为包含函数定义）
        assert code is not None
        assert "def kernel" in code

    def test_generate_build_script(self):
        """测试编译脚本生成"""
        request = AIKGRequest(
            fusion_name="Test Fusion",
            fusion_description="Test",
            operator_sequence=["Op1"],
            operator_pattern="Op1",
            input_shapes=[[100, 100]],
            output_shapes=[[100, 100]],
            data_types=["FP16"],
            target_speedup=1.5,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id="test",
            complexity="低",
            estimated_memory_saving=0.3,
        )

        script = self.client._generate_build_script(request)

        assert "#!/bin/bash" in script
        assert "test_fusion" in script.lower()
        assert "triton" in script
        assert "chmod" not in script  # 不应该在脚本中

    def test_generate_benchmark_code(self):
        """测试性能测试代码生成"""
        request = AIKGRequest(
            fusion_name="Test Fusion",
            fusion_description="Test",
            operator_sequence=["Op1"],
            operator_pattern="Op1",
            input_shapes=[[100, 100]],
            output_shapes=[[100, 100]],
            data_types=["FP16"],
            target_speedup=1.5,
            target_backend=AIKGBackend.ASCEND,
            original_opportunity_id="test",
            complexity="低",
            estimated_memory_saving=0.3,
        )

        code = self.client._generate_benchmark_code(request)

        assert "benchmark_fusion" in code
        assert "torch" in code
        assert "time.perf_counter" in code
        assert "1.5x" in code or "1.5" in code


class TestGeneratedKernel:
    """GeneratedKernel 测试"""

    def test_to_summary(self):
        """测试生成摘要"""
        kernel = GeneratedKernel(
            kernel_name="test_kernel",
            request_id="req123",
            triton_code="import triton",
            build_script="#!/bin/bash",
            benchmark_code="# benchmark",
            status=GenerationStatus.SUCCESS,
            estimated_speedup=1.5,
            actual_speedup=1.4,
        )

        summary = kernel.to_summary()

        assert summary["kernel_name"] == "test_kernel"
        assert summary["request_id"] == "req123"
        assert summary["status"] == "success"
        assert summary["estimated_speedup"] == 1.5
        assert summary["actual_speedup"] == 1.4
        assert summary["has_triton_code"] is True
        assert summary["has_build_script"] is True


class TestAIKGIntegrator:
    """AIKGIntegrator 集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.converter = AIKGRequestConverter(
            min_speedup_threshold=1.2,
            skip_native_ops=False
        )
        self.client = AIKGKernelClient()
        self.integrator = AIKGIntegrator(
            converter=self.converter,
            client=self.client
        )

    @pytest.mark.asyncio
    async def test_generate_from_opportunities_empty(self):
        """测试空融合机会列表"""
        kernels = await self.integrator.generate_from_opportunities([])

        assert len(kernels) == 0

    @pytest.mark.asyncio
    async def test_generate_from_opportunities_filtered(self):
        """测试所有融合机会被过滤"""
        opportunities = [
            FusionOpportunity(
                opportunity_type="fuse",
                name="Low Speedup",
                description="加速比太低",
                current_ops=[{"name": "Op1", "dur": 1000}],
                estimated_speedup=1.0,
                end_to_end_speedup=1.01,  # 低于阈值 1.2
                time_proportion=0.01,
                memory_saving=0.1,
                implementation="Test",
                complexity="低",
            )
        ]

        kernels = await self.integrator.generate_from_opportunities(opportunities)

        # 应该没有请求被生成（加速比太低）
        assert len(kernels) == 0

    @pytest.mark.asyncio
    async def test_generate_with_output_dir(self):
        """测试生成并保存到文件"""
        with TemporaryDirectory() as tmpdir:
            integrator = AIKGIntegrator(
                converter=self.converter,
                client=self.client,
                output_dir=Path(tmpdir)
            )

            opportunities = [
                FusionOpportunity(
                    opportunity_type="fuse",
                    name="Test Fusion",
                    description="Test",
                    current_ops=[{"name": "Op1", "dur": 10000}],
                    estimated_speedup=1.5,
                    end_to_end_speedup=1.25,  # 高于阈值
                    time_proportion=0.1,
                    memory_saving=0.3,
                    implementation="Test",
                    complexity="低",
                )
            ]

            kernels = await integrator.generate_from_opportunities(opportunities)

            # 由于没有配置 LLM 或服务，应该跳过
            assert len(kernels) == 1
            assert kernels[0].status == GenerationStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_generate_multiple_opportunities(self):
        """测试生成多个融合机会"""
        opportunities = [
            FusionOpportunity(
                opportunity_type="fuse",
                name=f"Fusion {i}",
                description=f"Test {i}",
                current_ops=[{"name": f"Op{i}", "dur": 10000 * (i + 1)}],
                estimated_speedup=1.3 + i * 0.1,
                end_to_end_speedup=1.25 + i * 0.05,  # 高于阈值
                time_proportion=0.1,
                memory_saving=0.3,
                implementation="Test",
                complexity="低",
            )
            for i in range(3)
        ]

        kernels = await self.integrator.generate_from_opportunities(opportunities)

        # 所有请求应该被跳过（因为没有配置 LLM/服务）
        assert len(kernels) == 3
        for k in kernels:
            assert k.status == GenerationStatus.SKIPPED


class TestAIKGIntegrationE2E:
    """端到端集成测试"""

    def test_full_conversion_pipeline(self):
        """测试完整的转换流程"""
        # 创建融合机会
        opportunities = [
            FusionOpportunity(
                opportunity_type="fuse",
                name="MatMul+Bias+Activation 融合",
                description="将矩阵乘法、偏置加法和激活函数融合",
                current_ops=[
                    {"name": "MatMulV2_1", "dur": 200000, "input_shapes": "[1024,1024];[1024,1024]"},
                    {"name": "BiasAdd_1", "dur": 10000, "input_shapes": "[1024,1024]"},
                    {"name": "GELU_1", "dur": 15000},
                ],
                estimated_speedup=1.3,
                end_to_end_speedup=1.25,  # 高于阈值
                time_proportion=0.18,
                memory_saving=0.4,
                implementation="使用 Triton 实现",
                complexity="中等",
            ),
            FusionOpportunity(
                opportunity_type="replace",
                name="FlashAttention 替换",
                description="昇腾已有算子",
                current_ops=[{"name": "Attention", "dur": 100000}],
                estimated_speedup=5.0,
                end_to_end_speedup=1.5,  # 高于阈值
                time_proportion=0.3,
                memory_saving=0.6,
                implementation="昇腾原生",
                complexity="低",
                ascend_op="aclnnFlashAttentionScore",
            ),
        ]

        # 创建转换器（不跳过原生算子）
        converter = AIKGRequestConverter(skip_native_ops=False)

        # 转换
        requests = converter.convert_opportunities(opportunities)

        # 验证
        assert len(requests) == 2

        # 第一个请求
        req1 = requests[0]
        assert "MatMul" in req1.fusion_name
        assert len(req1.operator_sequence) == 3
        assert req1.complexity == "中等"
        assert req1.target_speedup == 1.3

        # 第二个请求
        req2 = requests[1]
        assert "FlashAttention" in req2.fusion_name
        assert len(req2.operator_sequence) == 1
        assert req2.target_speedup == 5.0

        # 验证 prompt 生成
        prompt1 = req1.to_aikg_prompt()
        assert "MatMul+Bias+Activation" in prompt1
        assert "MatMulV2_1" in prompt1 or "MatMul" in prompt1
        assert "1.3x" in prompt1

        prompt2 = req2.to_aikg_prompt()
        assert "FlashAttention" in prompt2
        assert "5.0x" in prompt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
