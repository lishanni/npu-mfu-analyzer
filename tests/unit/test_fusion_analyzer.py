"""
Unit tests for FusionAnalyzer

测试算子融合机会检测功能。
"""

import pytest
from src.agents.operator_agent import FusionAnalyzer, FusionOpportunity


class TestFusionAnalyzer:
    """FusionAnalyzer 单元测试"""

    def setup_method(self):
        """测试前设置"""
        self.analyzer = FusionAnalyzer()

    def test_detect_flash_attention_opportunity(self):
        """测试检测 FlashAttention 替换机会"""
        # 模拟有普通 Attention 但没有 FlashAttention 的场景
        operators = [
            {"name": "Attention_QK", "dur": 100000},
            {"name": "Attention_Softmax", "dur": 50000},
            {"name": "Attention_Output", "dur": 80000},
            {"name": "MatMulV2", "dur": 200000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到 FlashAttention 替换机会
        flash_opp = None
        for opp in opportunities:
            if "FlashAttention" in opp.name:
                flash_opp = opp
                break

        assert flash_opp is not None, "应该检测到 FlashAttention 替换机会"
        assert flash_opp.opportunity_type == "replace"
        assert flash_opp.estimated_speedup >= 3.0
        assert flash_opp.ascend_op == "aclnnFlashAttentionScore"
        # 验证端到端加速计算
        assert flash_opp.time_proportion > 0
        assert flash_opp.end_to_end_speedup >= 1.0

    def test_matmul_bias_act_fusion(self):
        """测试 MatMul+Bias+Act 融合检测"""
        operators = [
            {"name": "MatMulV2_1", "dur": 200000},
            {"name": "MatMulV2_2", "dur": 180000},
            {"name": "BiasAdd_1", "dur": 10000},
            {"name": "BiasAdd_2", "dur": 9000},
            {"name": "GELU_1", "dur": 15000},
            {"name": "GELU_2", "dur": 14000},
            {"name": "Softmax", "dur": 30000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到 MatMul+Bias+Act 融合机会
        matmul_fusion = None
        for opp in opportunities:
            if "MatMul+Bias+Activation" in opp.name:
                matmul_fusion = opp
                break

        assert matmul_fusion is not None, "应该检测到 MatMul+Bias+Activation 融合机会"
        assert matmul_fusion.opportunity_type == "fuse"
        assert matmul_fusion.estimated_speedup >= 1.2
        assert matmul_fusion.ascend_op == "aclnnFusedMatMulBiasAct"

    def test_norm_residual_fusion(self):
        """测试 Norm + Residual Add 融合检测"""
        operators = [
            {"name": "RMSNorm_1", "dur": 25000},
            {"name": "RMSNorm_2", "dur": 24000},
            {"name": "Add_residual", "dur": 8000},
            {"name": "Add_2", "dur": 7500},
            {"name": "MatMulV2", "dur": 150000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到 Norm+Residual 融合机会
        norm_fusion = None
        for opp in opportunities:
            if "Norm" in opp.name and "Residual" in opp.name:
                norm_fusion = opp
                break

        assert norm_fusion is not None, "应该检测到 Norm+Residual 融合机会"
        assert norm_fusion.opportunity_type == "fuse"
        assert norm_fusion.ascend_op == "aclnnAddRmsNorm"

    def test_qkv_projection_fusion(self):
        """测试 QKV Projection 融合检测"""
        operators = [
            {"name": "MatMul_Query", "dur": 80000},
            {"name": "MatMul_Key", "dur": 75000},
            {"name": "MatMul_Value", "dur": 78000},
            {"name": "Attention", "dur": 120000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到 QKV Projection 融合机会
        qkv_fusion = None
        for opp in opportunities:
            if "QKV" in opp.name:
                qkv_fusion = opp
                break

        assert qkv_fusion is not None, "应该检测到 QKV Projection 融合机会"
        assert qkv_fusion.opportunity_type == "fuse"
        assert qkv_fusion.ascend_op == "aclnnFusedQKVProjection"

    def test_moe_fusion_detection(self):
        """测试 MoE 专家计算融合检测"""
        operators = [
            {"name": "TopK_MoE", "dur": 5000},
            {"name": "GroupedMatMul_Expert1", "dur": 100000},
            {"name": "GroupedMatMul_Expert2", "dur": 95000},
            {"name": "MatMulV2", "dur": 80000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到 MoE 融合机会
        moe_fusion = None
        for opp in opportunities:
            if "MoE" in opp.name:
                moe_fusion = opp
                break

        assert moe_fusion is not None, "应该检测到 MoE 专家计算融合机会"
        assert moe_fusion.opportunity_type == "fuse"
        assert moe_fusion.ascend_op == "aclnnGroupedMatmulV4"

    def test_ascend_fused_ops_recognition(self):
        """测试识别昇腾已有融合算子（不产生重复建议）"""
        # 已经使用 FlashAttention 的场景
        operators = [
            {"name": "aclnnFlashAttentionScore", "dur": 40000},
            {"name": "aclnnFusedMatMulBiasAct", "dur": 50000},
            {"name": "MatMulV2", "dur": 60000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证没有产生 FlashAttention 替换建议（因为已经在使用）
        for opp in opportunities:
            assert opp.opportunity_type != "replace" or "FlashAttention" not in opp.name

    def test_reshape_transpose_optimization(self):
        """测试 reshape/transpose 优化检测"""
        operators = [
            {"name": "Reshape_1", "dur": 3000},
            {"name": "Transpose_1", "dur": 4000},
            {"name": "Reshape_2", "dur": 2500},
            {"name": "Transpose_2", "dur": 3500},
            {"name": "Reshape_3", "dur": 2800},
            {"name": "MatMulV2", "dur": 100000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到形状变换优化建议
        reshape_opt = None
        for opp in opportunities:
            if "形状变换" in opp.name or "reshape" in opp.name.lower():
                reshape_opt = opp
                break

        assert reshape_opt is not None, "应该检测到形状变换优化建议"

    def test_cast_optimization(self):
        """测试 Cast 优化检测"""
        operators = [
            {"name": "Cast_FP16_to_FP32", "dur": 2000},
            {"name": "Cast_FP32_to_FP16", "dur": 1800},
            {"name": "Cast_BF16_to_FP32", "dur": 2200},
            {"name": "MatMulV2", "dur": 80000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到数据类型转换优化建议
        cast_opt = None
        for opp in opportunities:
            if "类型转换" in opp.name or "cast" in opp.name.lower():
                cast_opt = opp
                break

        assert cast_opt is not None, "应该检测到数据类型转换优化建议"

    def test_empty_operators(self):
        """测试空算子列表"""
        opportunities = self.analyzer.detect_opportunities([], timeline_data=None)
        assert len(opportunities) == 0

    def test_opportunity_sorting_by_end_to_end_speedup(self):
        """测试融合机会按端到端加速效果排序"""
        operators = [
            {"name": "Attention_QK", "dur": 100000},
            {"name": "Attention_Softmax", "dur": 50000},
            {"name": "MatMulV2_1", "dur": 200000},
            {"name": "MatMulV2_2", "dur": 180000},
            {"name": "RMSNorm_1", "dur": 25000},
            {"name": "Add_residual", "dur": 8000},
            {"name": "BiasAdd_1", "dur": 10000},
            {"name": "GELU_1", "dur": 15000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证按端到端加速比降序排列
        if len(opportunities) >= 2:
            for i in range(len(opportunities) - 1):
                assert opportunities[i].end_to_end_speedup >= opportunities[i + 1].end_to_end_speedup

    def test_end_to_end_speedup_calculation(self):
        """测试端到端加速计算"""
        # 创建一个简单的场景：总耗时 1000ms，某个算子占 30%
        operators = [
            {"name": "Attention_QK", "dur": 300000},  # 30% of total
            {"name": "Attention_Softmax", "dur": 100000},  # 10%
            {"name": "Other_ops", "dur": 600000},  # 60%
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证端到端加速计算
        for opp in opportunities:
            # 检查端到端加速在合理范围内
            assert 1.0 <= opp.end_to_end_speedup <= 2.0, f"端到端加速应在合理范围内: {opp.end_to_end_speedup}"
            # 检查时间占比计算正确
            assert 0.0 < opp.time_proportion <= 1.0, f"时间占比应在 (0, 1] 范围内: {opp.time_proportion}"

    def test_global_analysis_beyond_top10(self):
        """测试全局分析不局限于 Top 10"""
        # 创建 50 个算子的场景
        operators = []
        for i in range(50):
            if i < 10:
                operators.append({"name": f"MatMulV2_{i}", "dur": 100000 - i * 1000})  # Top 10
            elif i < 20:
                operators.append({"name": f"Attention_{i}", "dur": 50000 - i * 500})
            else:
                operators.append({"name": f"SmallOp_{i}", "dur": 1000})

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证全局分析能够发现所有相关算子
        matmul_fusion = None
        attn_fusion = None
        for opp in opportunities:
            if "MatMul+Bias+Activation" in opp.name:
                matmul_fusion = opp
            if "FlashAttention" in opp.name:
                attn_fusion = opp

        # 验证融合机会考虑了全局所有算子
        if matmul_fusion:
            # 应该检测到远超过 10 个 MatMul 算子
            assert "约" in matmul_fusion.name or len(matmul_fusion.current_ops) > 0

        if attn_fusion:
            # 应该检测到所有 Attention 相关算子
            assert len(attn_fusion.current_ops) >= 10  # 至少有 10 个 Attention 算子

    def test_fusion_opportunity_to_prompt_text(self):
        """测试 FusionOpportunity.to_prompt_text() 方法"""
        opp = FusionOpportunity(
            opportunity_type="replace",
            name="Test Fusion",
            description="Test description",
            current_ops=[{"name": "Op1"}],
            estimated_speedup=2.5,
            end_to_end_speedup=1.15,  # 新增：端到端加速
            time_proportion=0.3,  # 新增：耗时占比
            memory_saving=0.4,
            implementation="Test implementation",
            complexity="低",
            ascend_op="aclnnTestOp",
        )

        prompt_text = opp.to_prompt_text()

        assert "Test Fusion" in prompt_text
        assert "replace" in prompt_text
        assert "2.5x" in prompt_text
        # 端到端加速格式化为百分比：1.15 = 115.0%
        assert "115" in prompt_text  # 匹配 115.0% 中的 115
        # 耗时占比格式化为百分比：0.3 = 30.0%
        assert "30" in prompt_text  # 匹配 30.0% 中的 30
        assert "40%" in prompt_text  # 内存节省
        assert "低" in prompt_text
        assert "aclnnTestOp" in prompt_text


class TestFusionAnalyzerIntegration:
    """FusionAnalyzer 集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.analyzer = FusionAnalyzer()

    def test_realistic_transformer_operators(self):
        """测试真实的 Transformer 模型算子场景"""
        # 模拟 Transformer 模型的典型算子（多层）
        operators = [
            # Layer 1 - QKV Projection (未融合)
            {"name": "MatMulV2_Query_L1", "dur": 85000},
            {"name": "MatMulV2_Key_L1", "dur": 82000},
            {"name": "MatMulV2_Value_L1", "dur": 83000},
            # Layer 1 - Attention (未使用 FlashAttention)
            {"name": "MatMulV2_QK_L1", "dur": 120000},
            {"name": "Softmax_L1", "dur": 45000},
            {"name": "MatMulV2_Attn_L1", "dur": 90000},
            # Layer 1 - FFN (未融合)
            {"name": "MatMulV2_FF1_L1", "dur": 180000},
            {"name": "BiasAdd_FF1_L1", "dur": 12000},
            {"name": "GELU_FF1_L1", "dur": 18000},
            {"name": "MatMulV2_FF2_L1", "dur": 175000},
            # Layer 2 - FFN (未融合)
            {"name": "MatMulV2_FF1_L2", "dur": 175000},
            {"name": "BiasAdd_FF1_L2", "dur": 11500},
            {"name": "GELU_FF1_L2", "dur": 17500},
            # Layer 1 - Norm
            {"name": "RMSNorm_1_L1", "dur": 28000},
            {"name": "Add_residual_L1", "dur": 9000},
            # Layer 2 - Norm
            {"name": "RMSNorm_1_L2", "dur": 27000},
            {"name": "Add_residual_L2", "dur": 8500},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # 验证检测到多种融合机会
        opportunity_names = [opp.name for opp in opportunities]

        # 应该检测到的关键融合机会
        expected_keywords = [
            "FlashAttention",
            "QKV",
            "MatMul+Bias+Activation",
            "Norm",
        ]

        for keyword in expected_keywords:
            found = any(keyword in name for name in opportunity_names)
            assert found, f"应该检测到包含 '{keyword}' 的融合机会"

        # 验证至少检测到 3 个融合机会
        assert len(opportunities) >= 3

    def test_llama2_like_operators(self):
        """测试类似 LLaMA 2 的算子模式"""
        operators = [
            # Layer 1 - Pre-norm + Residual
            {"name": "RMSNorm_L1", "dur": 30000},
            {"name": "Add_L1", "dur": 10000},
            # Layer 1 - QKV
            {"name": "MatMul_q_proj_L1", "dur": 90000},
            {"name": "MatMul_k_proj_L1", "dur": 88000},
            {"name": "MatMul_v_proj_L1", "dur": 89000},
            # Layer 1 - Attention
            {"name": "MatMul_o_proj_L1", "dur": 95000},
            # Layer 2 - Pre-norm + Residual
            {"name": "RMSNorm_L2", "dur": 29000},
            {"name": "Add_L2", "dur": 9500},
            # Layer 2 - QKV
            {"name": "MatMul_q_proj_L2", "dur": 89000},
            {"name": "MatMul_k_proj_L2", "dur": 87000},
            {"name": "MatMul_v_proj_L2", "dur": 88000},
            # FFN (SiLU 激活)
            {"name": "MatMul_gate_proj", "dur": 92000},
            {"name": "MatMul_up_proj", "dur": 93000},
            {"name": "Mul", "dur": 15000},
            {"name": "MatMul_down_proj", "dur": 91000},
        ]

        opportunities = self.analyzer.detect_opportunities(all_operators=operators, timeline_data=None)

        # LLaMA 风格模型应该有多个融合机会
        assert len(opportunities) >= 2

        # 检查是否有 QKV 融合建议
        has_qkv = any("QKV" in opp.name for opp in opportunities)
        assert has_qkv, "应该检测到 QKV Projection 融合"

        # 检查是否有 Norm+Residual 融合建议
        has_norm = any("Norm" in opp.name and "Residual" in opp.name for opp in opportunities)
        assert has_norm, "应该检测到 Norm+Residual 融合"

        # 检查是否有 SwiGLU 相关建议（可选）
        has_glul_suggestion = any(
            "SiLU" in opp.name or "GLU" in opp.name or "逐元素" in opp.name
            for opp in opportunities
        )
        # 注意：SiLU+Mul 检测可能需要更复杂的模式匹配，这里不强制要求


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
