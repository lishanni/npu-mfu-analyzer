"""
集群拓扑与通信瓶颈诊断 - 集成测试

验证以下功能模块：
- TopologyAnalyzer: 物理拓扑分析、带宽利用率、慢链路识别
- CollectiveProfiler: 集合通信分析、带宽效率、算法推荐
- JitterDetector: 计算/通信/对齐抖动检测、慢 rank 识别
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径（conftest.py 会自动处理，此处作为独立运行的备份）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import (
    TopologyAnalyzer,
    CollectiveProfiler,
    CollectiveOpType,
)
from src.agents.jitter_agent import JitterDetector, JitterMetrics


def test_topology_analyzer():
    """测试拓扑分析器"""
    print("=" * 60)
    print("测试 1: 拓扑分析器 (TopologyAnalyzer)")
    print("=" * 60)
    
    # 测试用例 1: 单机 8 卡
    print("\n测试用例 1: 单机 8 卡")
    analyzer = TopologyAnalyzer(
        world_size=8,
        npus_per_machine=8,
        hccs_bandwidth=56.0,
        rdma_bandwidth=25.0,
    )
    
    topology = analyzer.build_topology()
    print(f"  机器数: {topology.num_machines}")
    print(f"  每机器 NPU 数: {topology.npus_per_machine}")
    print(f"  节点数: {len(topology.nodes)}")
    print(f"  链路数: {len(topology.links)}")
    print(f"  节点内带宽: {topology.intra_node_bandwidth} GB/s")
    print(f"  节点间带宽: {topology.inter_node_bandwidth} GB/s")
    
    # 模拟通信事件
    comm_events = [
        {
            "src_rank": 0,
            "dst_rank": 1,
            "data_size": 1024 * 1024 * 100,  # 100 MB
            "duration": 2000,  # 2ms
        },
        {
            "src_rank": 0,
            "dst_rank": 2,
            "data_size": 1024 * 1024 * 100,
            "duration": 2500,
        },
    ]
    
    metrics = analyzer.analyze(comm_events)
    print(f"\n  带宽利用率:")
    print(f"    节点内: {metrics.intra_node_bw_utilization:.1%}")
    print(f"    节点间: {metrics.inter_node_bw_utilization:.1%}")
    print(f"  平均链路带宽: {metrics.avg_link_bandwidth:.2f} GB/s")
    
    # 测试用例 2: 2 机 16 卡
    print("\n测试用例 2: 2 机 16 卡")
    analyzer2 = TopologyAnalyzer(
        world_size=16,
        npus_per_machine=8,
    )
    
    topology2 = analyzer2.build_topology()
    print(f"  机器数: {topology2.num_machines}")
    print(f"  Machine 0 ranks: {topology2.get_machine_ranks(0)}")
    print(f"  Machine 1 ranks: {topology2.get_machine_ranks(1)}")
    
    # 测试跨节点通信路径
    path_info = analyzer2.get_communication_path(0, 8)
    print(f"\n  Rank 0 -> Rank 8 通信路径:")
    print(f"    同一机器: {path_info['is_same_machine']}")
    print(f"    链路类型: {path_info['link_type']}")
    print(f"    理论带宽: {path_info['theoretical_bandwidth']} GB/s")


def test_collective_profiler():
    """测试集合通信分析器"""
    print("\n" + "=" * 60)
    print("测试 2: 集合通信分析器 (CollectiveProfiler)")
    print("=" * 60)
    
    profiler = CollectiveProfiler(theoretical_bandwidth_gbps=56.0)
    
    # 模拟集合操作事件
    comm_events = [
        # AllReduce 事件
        {
            "name": "AllReduce",
            "dur": 5000,  # 5ms
            "args": {
                "data_size": 1024 * 1024 * 400,  # 400 MB
                "group_size": 8,
                "group_name": "dp_group",
            }
        },
        # ReduceScatter 事件
        {
            "name": "ReduceScatter",
            "dur": 3000,
            "args": {
                "data_size": 1024 * 1024 * 200,
                "group_size": 8,
                "group_name": "zero_group",
            }
        },
        # AllGather 事件
        {
            "name": "AllGather",
            "dur": 3500,
            "args": {
                "data_size": 1024 * 1024 * 200,
                "group_size": 8,
                "group_name": "zero_group",
            }
        },
        # 小数据量 AllReduce
        {
            "name": "AllReduce",
            "dur": 500,
            "args": {
                "data_size": 1024 * 10,  # 10 KB
                "group_size": 8,
                "group_name": "tp_group",
            }
        },
    ]
    
    result = profiler.profile(comm_events)
    
    print(f"\n  总体概况:")
    print(f"    总通信时间: {result.total_comm_time_ms:.2f} ms")
    print(f"    总数据量: {result.total_data_volume_gb:.4f} GB")
    print(f"    平均带宽效率: {result.avg_bandwidth_efficiency:.1%}")
    
    print(f"\n  各操作类型时间占比:")
    for op_type, ratio in sorted(
        result.time_by_op_type.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"    {op_type}: {ratio:.1%}")
    
    if result.bottleneck_ops:
        print(f"\n  瓶颈操作 (Top 3):")
        for op in result.bottleneck_ops[:3]:
            print(f"    {op.name}: {op.duration_us/1000:.2f}ms, "
                  f"带宽 {op.achieved_bandwidth_gbps:.2f} GB/s, "
                  f"效率 {op.bandwidth_efficiency:.1%}")
    
    if result.inefficient_ops:
        print(f"\n  低效操作:")
        for op in result.inefficient_ops[:3]:
            print(f"    {op.name}: 效率仅 {op.bandwidth_efficiency:.1%}")
    
    # 测试算法推荐
    print(f"\n  算法推荐:")
    for data_size, label in [(1024 * 100, "100KB"), (1024 * 1024 * 10, "10MB")]:
        algo = profiler.estimate_optimal_algorithm(
            CollectiveOpType.ALLREDUCE,
            data_size,
            8
        )
        print(f"    数据量 {label}: 推荐 {algo.value} 算法")


def test_jitter_detector():
    """测试抖动检测器"""
    print("\n" + "=" * 60)
    print("测试 3: 抖动检测器 (JitterDetector)")
    print("=" * 60)
    
    detector = JitterDetector()
    
    # 测试用例 1: 计算抖动
    print("\n测试用例 1: 计算抖动检测")
    
    # 模拟算子执行轨迹（有抖动）
    np.random.seed(42)
    operator_traces = {
        "MatMul_0": [100 + np.random.normal(0, 15) for _ in range(20)],  # 均值 100, std 15
        "LayerNorm_0": [50 + np.random.normal(0, 5) for _ in range(20)],
        "Softmax_0": [30 + np.random.normal(0, 10) for _ in range(20)],  # 高抖动
    }
    
    std, cv, outliers = detector.detect_compute_jitter(operator_traces)
    print(f"  标准差: {std:.2f} us")
    print(f"  变异系数 (CV): {cv:.1%}")
    print(f"  异常值数量: {outliers}")
    
    if cv > detector.CV_THRESHOLD:
        print(f"  ⚠️  检测到计算抖动 (CV > {detector.CV_THRESHOLD:.0%})")
    else:
        print(f"  ✓ 计算稳定")
    
    # 测试用例 2: 通信抖动
    print("\n测试用例 2: 通信抖动检测")
    
    comm_events = [
        {"name": "AllReduce", "dur": 5000 + np.random.normal(0, 800)}
        for _ in range(15)
    ]
    
    comm_std, comm_cv, comm_outliers = detector.detect_communication_jitter(comm_events)
    print(f"  标准差: {comm_std:.2f} us")
    print(f"  变异系数 (CV): {comm_cv:.1%}")
    print(f"  异常值数量: {comm_outliers}")
    
    if comm_cv > detector.CV_THRESHOLD:
        print(f"  ⚠️  检测到通信抖动 (CV > {detector.CV_THRESHOLD:.0%})")
    
    # 测试用例 3: 跨 rank 抖动
    print("\n测试用例 3: 跨 rank 抖动检测")
    
    # 模拟各 rank 的 step 耗时（rank 3 和 rank 5 较慢）
    rank_durations = {}
    for rank in range(8):
        base_time = 1000
        if rank in [3, 5]:
            base_time = 1200  # 慢 20%
        rank_durations[rank] = [
            base_time + np.random.normal(0, 50)
            for _ in range(10)
        ]
    
    variance, slow_ranks = detector.detect_cross_rank_jitter(rank_durations)
    print(f"  跨 rank 方差: {variance:.2f}")
    print(f"  慢 rank: {slow_ranks}")
    
    # 测试用例 4: 对齐抖动
    print("\n测试用例 4: 对齐抖动检测")
    
    compute_events = [
        {"ts": 1000 + i * 100, "dur": 80}
        for i in range(10)
    ]
    comm_events_align = [
        {"ts": 1000 + i * 100 + 85 + np.random.uniform(0, 20), "dur": 10}
        for i in range(10)
    ]
    
    max_skew, avg_skew = detector.detect_alignment_jitter(
        compute_events, comm_events_align
    )
    print(f"  最大偏差: {max_skew:.2f} us")
    print(f"  平均偏差: {avg_skew:.2f} us")
    
    if max_skew > 1000:
        print(f"  ⚠️  对齐偏差较大")
    
    # 测试用例 5: 根因分析
    print("\n测试用例 5: 根因分析")
    
    metrics = JitterMetrics(
        compute_jitter_std=std,
        compute_jitter_cv=cv,
        compute_outliers=outliers,
        comm_jitter_std=comm_std,
        comm_jitter_cv=comm_cv,
        comm_outliers=comm_outliers,
        alignment_skew_max=max_skew,
        alignment_skew_avg=avg_skew,
        cross_rank_variance=variance,
        slow_ranks=slow_ranks,
    )
    
    causes = detector.analyze_root_causes(metrics)
    print(f"  可能原因:")
    for cause in causes:
        print(f"    - {cause}")


def test_integration():
    """测试集成功能"""
    print("\n" + "=" * 60)
    print("测试 4: 集成测试")
    print("=" * 60)
    
    # 模拟完整的分析流程
    print("\n场景: 2 机 16 卡训练，检测拓扑、通信和抖动问题")
    
    # 1. 拓扑分析
    print("\n1. 拓扑分析")
    analyzer = TopologyAnalyzer(world_size=16, npus_per_machine=8)
    topology = analyzer.build_topology()
    
    # 模拟跨节点通信（带宽较低）
    comm_events_topo = []
    for i in range(8):
        for j in range(8, 16):
            comm_events_topo.append({
                "src_rank": i,
                "dst_rank": j,
                "data_size": 1024 * 1024 * 50,
                "duration": 3000,  # 较慢的跨节点通信
            })
    
    topo_metrics = analyzer.analyze(comm_events_topo)
    print(f"  节点间带宽利用率: {topo_metrics.inter_node_bw_utilization:.1%}")
    if topo_metrics.bandwidth_bottleneck:
        print(f"  ⚠️  瓶颈: {topo_metrics.bandwidth_bottleneck}")
    
    # 2. 集合通信分析
    print("\n2. 集合通信分析")
    profiler = CollectiveProfiler()
    comm_events_collective = [
        {
            "name": "AllReduce",
            "dur": 8000,
            "args": {"data_size": 1024 * 1024 * 500, "group_size": 16}
        }
    ] * 5
    
    coll_result = profiler.profile(comm_events_collective)
    print(f"  总通信时间: {coll_result.total_comm_time_ms:.2f} ms")
    print(f"  平均带宽效率: {coll_result.avg_bandwidth_efficiency:.1%}")
    
    # 3. 抖动检测
    print("\n3. 抖动检测")
    detector = JitterDetector()
    
    rank_durations = {rank: [1000 + np.random.normal(0, 100) for _ in range(10)] 
                      for rank in range(16)}
    variance, slow_ranks = detector.detect_cross_rank_jitter(rank_durations)
    
    print(f"  跨 rank 方差: {variance:.2f}")
    if slow_ranks:
        print(f"  检测到慢 rank: {slow_ranks}")
    
    print("\n✓ 集成测试完成")


if __name__ == "__main__":
    print("Phase 7: 集群拓扑与通信瓶颈诊断 - 功能测试")
    print("=" * 60)
    
    test_topology_analyzer()
    test_collective_profiler()
    test_jitter_detector()
    test_integration()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
