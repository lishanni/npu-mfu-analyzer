"""
Skill Engine 集成测试

验证 Phase 8 的专家技能引擎功能
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.skills import (
    SkillEngine,
    get_engine,
    get_registry,
    SkillCategory,
    SkillResult,
    LogicChain,
)
from src.skills.python_skills import register_all_skills
from src.skills.prompts import register_all_prompt_skills


def test_skill_registration():
    """测试技能注册"""
    print("=" * 60)
    print("测试 1: 技能注册")
    print("=" * 60)
    
    registry = get_registry()
    registry.clear()  # 清空之前的注册
    
    # 注册 Python 技能
    python_count = register_all_skills()
    print(f"\n注册 Python 技能: {python_count} 个")
    
    # 注册 Prompt 技能
    prompt_count = register_all_prompt_skills()
    print(f"注册 Prompt 技能: {prompt_count} 个")
    
    # 列出所有技能
    print(f"\n总计: {registry.total_skill_count} 个技能")
    
    # 按分类列出
    print("\n按分类统计:")
    for category in SkillCategory:
        skills = registry.list_python_skills(category=category)
        if skills:
            print(f"  {category.value}: {len(skills)} 个")
            for skill in skills[:2]:  # 只显示前 2 个
                print(f"    - {skill.metadata.name}: {skill.metadata.description[:40]}...")


def test_mfu_skill():
    """测试 MFU 计算技能"""
    print("\n" + "=" * 60)
    print("测试 2: MFU 计算技能")
    print("=" * 60)
    
    engine = get_engine()
    
    # 测试场景：7B 模型在 Atlas A2 上训练
    result = engine.execute_skill(
        "calculate_mfu",
        model_flops=2e15,  # 2 PFLOPS per step
        step_time_ms=500,  # 500ms per step
        peak_tflops=280,   # Atlas A2
        num_gpus=8,
        precision="bf16",
    )
    
    print(f"\n执行结果: {'成功' if result.success else '失败'}")
    print(f"执行时间: {result.execution_time_ms:.2f}ms")
    print(f"\n{result.to_prompt_text()}")


def test_bandwidth_skill():
    """测试带宽效率检查技能"""
    print("\n" + "=" * 60)
    print("测试 3: 带宽效率检查技能")
    print("=" * 60)
    
    engine = get_engine()
    
    # 测试 HCCS 带宽
    result = engine.execute_skill(
        "check_bandwidth_efficiency",
        measured_bandwidth_gbps=45.0,
        theoretical_bandwidth_gbps=56.0,
        bandwidth_type="hccs",
        data_size_mb=100,
    )
    
    print(f"\n执行结果: {'成功' if result.success else '失败'}")
    print(f"\n{result.to_prompt_text()}")


def test_overlap_skill():
    """测试通信掩盖率技能"""
    print("\n" + "=" * 60)
    print("测试 4: 通信掩盖率检查技能")
    print("=" * 60)
    
    engine = get_engine()
    
    # 测试场景：70% 掩盖率
    result = engine.execute_skill(
        "check_overlap_ratio",
        total_compute_time_us=800000,  # 800ms
        total_comm_time_us=200000,     # 200ms
        overlapped_time_us=140000,     # 140ms 被掩盖
        free_time_us=10000,            # 10ms 空闲
    )
    
    print(f"\n执行结果: {'成功' if result.success else '失败'}")
    print(f"\n{result.to_prompt_text()}")


def test_slow_rank_skill():
    """测试慢卡检测技能"""
    print("\n" + "=" * 60)
    print("测试 5: 慢卡检测技能")
    print("=" * 60)
    
    engine = get_engine()
    
    # 模拟 8 卡，其中 rank 3 和 rank 7 较慢
    rank_times = {
        0: 1000.0,
        1: 1010.0,
        2: 995.0,
        3: 1250.0,  # 慢
        4: 1005.0,
        5: 998.0,
        6: 1002.0,
        7: 1180.0,  # 慢
    }
    
    result = engine.execute_skill(
        "detect_slow_rank",
        rank_times=rank_times,
        method="three_sigma",
        threshold_sigma=1.5,
    )
    
    print(f"\n执行结果: {'成功' if result.success else '失败'}")
    print(f"\n{result.to_prompt_text()}")


def test_jitter_skill():
    """测试抖动检测技能"""
    print("\n" + "=" * 60)
    print("测试 6: 抖动检测技能")
    print("=" * 60)
    
    engine = get_engine()
    
    # 模拟有抖动的算子执行时间
    import random
    random.seed(42)
    durations = [100 + random.gauss(0, 20) for _ in range(30)]
    
    result = engine.execute_skill(
        "detect_compute_jitter",
        durations=durations,
        operator_name="MatMul",
        cv_threshold=0.15,
    )
    
    print(f"\n执行结果: {'成功' if result.success else '失败'}")
    print(f"\n{result.to_prompt_text()}")


def test_prompt_skill():
    """测试 Prompt 技能"""
    print("\n" + "=" * 60)
    print("测试 7: Prompt 技能渲染")
    print("=" * 60)
    
    engine = get_engine()
    
    # 渲染诊断流程 prompt
    prompt = engine.render_prompt_skill(
        "diagnosis_flow",
        target_mfu=50,
    )
    
    if prompt:
        print("\n诊断流程 Prompt (前 500 字符):")
        print("-" * 40)
        print(prompt[:500] + "...")
    else:
        print("Prompt 技能渲染失败")


def test_logic_chain():
    """测试逻辑链执行"""
    print("\n" + "=" * 60)
    print("测试 8: 逻辑链执行")
    print("=" * 60)
    
    engine = get_engine()
    
    # 创建自定义逻辑链
    chain = engine.build_chain(
        "test_chain",
        "测试用逻辑链"
    ).add_step(
        "calculate_mfu",
        inputs={
            "model_flops": 1e15,
            "step_time_ms": 400,
            "peak_tflops": 280,
            "num_gpus": 4,
        }
    ).add_step(
        "check_overlap_ratio",
        inputs={
            "total_compute_time_us": 300000,
            "total_comm_time_us": 100000,
            "overlapped_time_us": 70000,
        }
    )
    
    # 执行链
    result = engine.execute_chain("test_chain")
    
    print(f"\n逻辑链状态: {result.status.value}")
    print(f"总耗时: {result.total_time_ms:.2f}ms")
    print(f"执行步骤: {len(result.steps)} 个")
    
    for i, step in enumerate(result.steps, 1):
        status = "✓" if step.result and step.result.success else "✗"
        print(f"  {i}. {step.skill_name} [{status}]")


def test_skill_search():
    """测试技能搜索"""
    print("\n" + "=" * 60)
    print("测试 9: 技能搜索")
    print("=" * 60)
    
    registry = get_registry()
    
    # 搜索关键词
    keywords = ["mfu", "bandwidth", "jitter", "slow"]
    
    for keyword in keywords:
        results = registry.search_skills(keyword)
        print(f"\n搜索 '{keyword}': {len(results)} 个结果")
        for skill in results[:2]:
            if hasattr(skill, 'metadata'):
                print(f"  - {skill.metadata.name}")
            else:
                print(f"  - {skill.name}")


def test_skill_catalog():
    """测试技能目录生成"""
    print("\n" + "=" * 60)
    print("测试 10: 技能目录生成")
    print("=" * 60)
    
    registry = get_registry()
    catalog = registry.get_skill_catalog()
    
    print("\n技能目录 (前 800 字符):")
    print("-" * 40)
    print(catalog[:800] + "...")


def main():
    """运行所有测试"""
    print("Phase 8: Skill Engine 功能测试")
    print("=" * 60)
    
    # 先注册所有技能
    registry = get_registry()
    registry.clear()
    register_all_skills()
    register_all_prompt_skills()
    
    # 运行测试
    test_skill_registration()
    test_mfu_skill()
    test_bandwidth_skill()
    test_overlap_skill()
    test_slow_rank_skill()
    test_jitter_skill()
    test_prompt_skill()
    test_logic_chain()
    test_skill_search()
    test_skill_catalog()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
