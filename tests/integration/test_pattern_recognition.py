"""
跨框架模式识别 - 集成测试

验证以下功能模块：
- FrameworkDetector: 训练框架检测 (Megatron/DeepSpeed/FSDP/MindSpeed)
- ParallelDetector: 并行策略检测 (TP/PP/DP/ZeRO/FSDP/CP/EP)
- ModelDetector: 模型结构检测 (layers/hidden_size/attention_heads)
- UniversalPatternMatcher: 统一模式匹配入口
"""

import sys
from pathlib import Path

# 添加项目路径（conftest.py 会自动处理，此处作为独立运行的备份）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pattern_matcher import (
    FrameworkDetector,
    ParallelDetector,
    ModelDetector,
    UniversalPatternMatcher,
)


def test_framework_detector():
    """测试框架检测器"""
    print("=" * 60)
    print("测试 1: 框架检测器 (FrameworkDetector)")
    print("=" * 60)
    
    detector = FrameworkDetector()
    
    # 测试用例 1: Megatron
    operator_names_megatron = [
        "megatron_parallel_embedding",
        "column_parallel_linear",
        "row_parallel_linear",
        "allreduce",
    ]
    comm_groups_megatron = ["tensor_model_parallel_group", "data_parallel_group"]
    
    result = detector.detect(operator_names_megatron, comm_groups_megatron)
    print(f"\n测试用例 1: Megatron")
    print(f"  检测结果: {result.framework.value}")
    print(f"  置信度: {result.confidence:.0%}")
    print(f"  证据: {result.evidence[:3]}")  # 显示前 3 条
    
    # 测试用例 2: DeepSpeed
    operator_names_deepspeed = [
        "deepspeed_adam",
        "zero_optimizer",
        "reducescatter",
        "allgather",
    ]
    comm_groups_deepspeed = ["deepspeed_group", "zero_stage_2"]
    
    result = detector.detect(operator_names_deepspeed, comm_groups_deepspeed)
    print(f"\n测试用例 2: DeepSpeed")
    print(f"  检测结果: {result.framework.value}")
    print(f"  置信度: {result.confidence:.0%}")
    print(f"  证据: {result.evidence[:3]}")
    
    # 测试用例 3: PyTorch DDP
    operator_names_ddp = [
        "distributed_data_parallel",
        "backward",
        "allreduce",
    ]
    comm_groups_ddp = ["default"]
    env_vars_ddp = {"MASTER_ADDR": "localhost", "RANK": "0", "WORLD_SIZE": "8"}
    
    result = detector.detect(operator_names_ddp, comm_groups_ddp, env_vars_ddp)
    print(f"\n测试用例 3: PyTorch DDP")
    print(f"  检测结果: {result.framework.value}")
    print(f"  置信度: {result.confidence:.0%}")
    print(f"  证据: {result.evidence[:3]}")


def test_parallel_detector():
    """测试并行配置检测器"""
    print("\n" + "=" * 60)
    print("测试 2: 并行配置检测器 (ParallelDetector)")
    print("=" * 60)
    
    # 测试用例 1: TP=2, DP=4 (world_size=8)
    detector = ParallelDetector(world_size=8)
    
    comm_events = [
        {
            "name": "allreduce",
            "args": {
                "group_name": "tp_group",
                "group_size": 2,
            }
        },
        {
            "name": "allreduce",
            "args": {
                "group_name": "dp_group",
                "group_size": 4,
            }
        },
    ]
    
    config = detector.detect(comm_events)
    print(f"\n测试用例 1: TP=2, DP=4")
    print(f"  World Size: {config.world_size}")
    print(f"  Tensor Parallel: {config.tensor_parallel_size}")
    print(f"  Data Parallel: {config.data_parallel_size}")
    print(f"  Pipeline Parallel: {config.pipeline_parallel_size}")
    print(f"  置信度: {config.confidence:.0%}")
    print(f"  证据: {config.evidence}")
    
    # 测试用例 2: PP=4 with P2P communication
    detector2 = ParallelDetector(world_size=4)
    
    comm_events_pp = [
        {"name": "send", "args": {"dst_rank": 1}},
        {"name": "recv", "args": {"src_rank": 0}},
        {"name": "send", "args": {"dst_rank": 2}},
        {"name": "recv", "args": {"src_rank": 1}},
    ]
    
    config2 = detector2.detect(comm_events_pp)
    print(f"\n测试用例 2: Pipeline Parallel")
    print(f"  World Size: {config2.world_size}")
    print(f"  Pipeline Parallel: {config2.pipeline_parallel_size}")
    print(f"  证据: {config2.evidence}")


def test_model_detector():
    """测试模型结构检测器"""
    print("\n" + "=" * 60)
    print("测试 3: 模型结构检测器 (ModelDetector)")
    print("=" * 60)
    
    detector = ModelDetector()
    
    # 测试用例 1: GPT-like 模型
    operator_names = [
        "gpt_layer.0.attention.qkv",
        "gpt_layer.1.attention.qkv",
        "gpt_layer.23.mlp.fc1",
        "layer_norm",
        "causal_attention",
    ]
    
    operator_shapes = {
        "gpt_layer.0.attention.qkv": [4096, 12288],  # hidden=4096, 3*hidden
        "gpt_layer.0.mlp.fc1": [4096, 16384],  # hidden -> intermediate
    }
    
    config = detector.detect(operator_names, operator_shapes)
    print(f"\n测试用例 1: GPT-like 模型")
    print(f"  架构: {config.architecture.value}")
    print(f"  层数: {config.num_layers}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  置信度: {config.confidence:.0%}")
    print(f"  证据: {config.evidence}")
    
    # 测试用例 2: LLaMA-like 模型
    operator_names_llama = [
        "llama_layer.0.attention",
        "llama_layer.1.attention",
        "llama_layer.31.mlp",
        "rotary_embedding",
        "rms_norm",
    ]
    
    config2 = detector.detect(operator_names_llama)
    print(f"\n测试用例 2: LLaMA-like 模型")
    print(f"  架构: {config2.architecture.value}")
    print(f"  层数: {config2.num_layers}")
    print(f"  置信度: {config2.confidence:.0%}")


def test_universal_matcher():
    """测试统一匹配器"""
    print("\n" + "=" * 60)
    print("测试 4: 统一模式匹配器 (UniversalPatternMatcher)")
    print("=" * 60)
    
    # 模拟一个简化的 ProfilingLoader
    class MockLoader:
        def detect(self):
            class Info:
                rank_count = 8
            return Info()
        
        def get_timeline_summary(self):
            return {
                "raw_data": [
                    {"name": "megatron_parallel_embedding"},
                    {"name": "column_parallel_linear"},
                ]
            }
        
        def get_overlap_events(self):
            return {
                "hccl": [
                    {
                        "name": "allreduce",
                        "args": {
                            "group_name": "tensor_model_parallel_group",
                            "group_size": 2,
                        }
                    },
                    {
                        "name": "allreduce",
                        "args": {
                            "group_name": "data_parallel_group",
                            "group_size": 4,
                        }
                    },
                ]
            }
    
    matcher = UniversalPatternMatcher()
    loader = MockLoader()
    
    try:
        pattern = matcher.detect(loader)
        
        print(f"\n综合检测结果:")
        print(f"\n框架:")
        print(f"  类型: {pattern.framework.framework.value}")
        print(f"  置信度: {pattern.framework.confidence:.0%}")
        
        print(f"\n并行配置:")
        print(f"  World Size: {pattern.parallel_config.world_size}")
        print(f"  TP: {pattern.parallel_config.tensor_parallel_size}")
        print(f"  DP: {pattern.parallel_config.data_parallel_size}")
        print(f"  PP: {pattern.parallel_config.pipeline_parallel_size}")
        
        print(f"\n模型:")
        print(f"  架构: {pattern.model_config.architecture.value}")
        
        print(f"\n生成 Prompt 文本:")
        print("-" * 60)
        print(pattern.to_prompt_text())
        
    except Exception as e:
        print(f"  测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Phase 6.2: Universal Pattern Matcher 功能测试")
    print("=" * 60)
    
    test_framework_detector()
    test_parallel_detector()
    test_model_detector()
    test_universal_matcher()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
