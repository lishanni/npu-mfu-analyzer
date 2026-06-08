"""
Skills 演示脚本

展示如何在 Claude Code 中使用 NPU MFU Analyzer Skills。
"""

from npu_mfu_analyzer.skills.v2.sdk import NPUMFUAnalyzerSDK
from npu_mfu_analyzer.skills.v2.base import SkillContext, SkillResult

# ============================================
# 方式 1: 使用 SDK 分析 Profiling 数据
# ============================================

def demo_sdk_usage():
    """演示 SDK 基本用法"""
    sdk = NPUMFUAnalyzerSDK()

    # 1. 列出所有可用技能
    print("=" * 60)
    print("📋 可用技能列表")
    print("=" * 60)
    skills = sdk.list_skills()
    for skill in skills:
        print(f"  • {skill['name']}: {skill['description'][:40]}...")

    print()

    # 2. 获取单个技能详情
    print("=" * 60)
    print("📊 技能详情: compute_mfu")
    print("=" * 60)
    info = sdk.get_skill("compute_mfu")
    print(f"  名称: {info['display_name']}")
    print(f"  类型: {info['skill_type']}")
    print(f"  描述: {info['description']}")


# ============================================
# 方式 2: 直接调用 Skill（需要 Profiling 数据）
# ============================================

def demo_direct_skill_call():
    """演示直接调用 Skill"""
    from npu_mfu_analyzer.skills.v2.skills.compute import MFUSkill

    # 创建 Skill 实例
    skill = MFUSkill()
    print(f"\n✅ 创建技能: {skill.name}")

    # 创建模拟上下文（实际使用时需要真实数据）
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1000000,  # 1秒
            "avg_compute_time": 600000,  # 600ms
            "avg_comm_time": 300000,  # 300ms
            "operators": [],
        }
    )

    # 执行技能
    result = skill.run(context)

    print(f"\n📊 执行结果:")
    print(f"  状态: {'✅ 成功' if result.success else '❌ 失败'}")
    print(f"  摘要: {result.summary}")
    if result.data:
        for k, v in result.data.items():
            print(f"  {k}: {v}")


# ============================================
# 方式 3: 执行技能链
# ============================================

def demo_skill_chain():
    """演示技能链执行"""
    from npu_mfu_analyzer.skills.v2.engine import SkillEngine
    from npu_mfu_analyzer.skills.v2.registry import SkillRegistry
    from npu_mfu_analyzer.skills.v2.skills.compute import MFUSkill, BandwidthSkill

    # 注册技能
    registry = SkillRegistry()
    registry.register(MFUSkill())
    registry.register(BandwidthSkill())

    # 创建引擎
    engine = SkillEngine(registry)

    print("\n" + "=" * 60)
    print("🔗 技能链执行")
    print("=" * 60)

    # 创建上下文
    context = SkillContext(
        profiling_summary={
            "avg_step_time": 1000000,
            "avg_compute_time": 600000,
            "avg_comm_time": 300000,
            "avg_comm_time": 200000,
            "total_comm_data_mb": 500,
            "operators": [],
        }
    )

    # 执行多个技能
    skills_to_run = ["compute_mfu", "compute_bandwidth"]
    results = engine.execute_skills(skills_to_run, context)

    for name, result in results.items():
        print(f"\n📌 {name}:")
        print(f"   {result.summary}")


# ============================================
# 方式 4: Claude Code 直接调用（通过 Bash）
# ============================================

def print_claude_usage():
    """打印 Claude Code 使用说明"""
    print("\n" + "=" * 60)
    print("🤖 Claude Code 调用方式")
    print("=" * 60)

    print("""
在 Claude Code 中，你可以这样调用技能：

1️⃣ 列出技能:
   python -m npu_mfu_analyzer.skills.v2.skill_cli list-skills

2️⃣ 查看技能详情:
   python -m npu_mfu_analyzer.skills.v2.skill_cli skill-info compute_mfu

3️⃣ 分析 Profiling 数据:
   python -m npu_mfu_analyzer.skills.v2.skill_cli analyze /path/to/profiling

4️⃣ 对比两个 Profiling:
   python -m npu_mfu_analyzer.skills.v2.skill_cli compare /path/to/baseline /path/to/target

5️⃣ 执行单个技能:
   python -m npu_mfu_analyzer.skills.v2.skill_cli execute compute_mfu /path/to/profiling
""")


if __name__ == "__main__":
    demo_sdk_usage()
    demo_direct_skill_call()
    demo_skill_chain()
    print_claude_usage()