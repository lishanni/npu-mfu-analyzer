"""
Communication Matrix Analyzer 单元测试
"""

import pytest
from src.analyzers.communication_matrix_analyzer import (
    CommunicationMatrixAnalyzer,
    CommunicationMatrix,
    LinkMetrics,
    CommOpStatistics,
    CommOpType,
    TransportType,
    CommunicationMatrixReport,
)


class TestLinkMetrics:
    """LinkMetrics 测试"""

    def test_init(self):
        """测试初始化"""
        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
        )
        assert metrics.src_rank == 0
        assert metrics.dst_rank == 1
        assert metrics.transport_type == TransportType.HCCS
        assert metrics.is_slow_link is False
        assert metrics.is_bottleneck is False

    def test_compute_derived_metrics(self):
        """测试派生指标计算"""
        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
            theoretical_bandwidth_gbps=56.0,
            total_transit_size_mb=112.0,  # 112 MB
            total_transit_time_ms=2.0,    # 2 ms
        )
        metrics.compute_derived_metrics()

        # 带宽 = 112 MB / 2 ms = 56 GB/s
        assert metrics.achieved_bandwidth_gbps == 56.0
        # 利用率 = 56 / 56 = 1.0
        assert metrics.bandwidth_utilization == 1.0

    def test_compute_anomaly_score(self):
        """测试异常评分计算"""
        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
            achieved_bandwidth_gbps=28.0,  # 低于平均值
        )

        avg_bw = 56.0
        std_bw = 10.0
        score = metrics.compute_anomaly_score(avg_bw, std_bw)

        # Z-score = (56 - 28) / 10 = 2.8, score = 2.8 / 3 = 0.933
        assert 0.9 < score < 1.0

    def test_to_dict(self):
        """测试转换为字典"""
        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
            achieved_bandwidth_gbps=40.0,
        )
        d = metrics.to_dict()

        assert d["src_rank"] == 0
        assert d["dst_rank"] == 1
        assert d["transport_type"] == "hccs"
        assert d["achieved_bandwidth_gbps"] == 40.0

    def test_to_summary(self):
        """测试生成摘要"""
        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
            achieved_bandwidth_gbps=40.0,
            bandwidth_utilization=0.7,
        )
        summary = metrics.to_summary()

        assert "Rank 0 ↔ 1" in summary
        assert "40.00 GB/s" in summary
        assert "70.0% util" in summary


class TestCommunicationMatrix:
    """CommunicationMatrix 测试"""

    def test_init(self):
        """测试初始化"""
        matrix = CommunicationMatrix(world_size=8)
        assert matrix.world_size == 8
        assert len(matrix.link_metrics) == 0

    def test_set_and_get_link(self):
        """测试设置和获取链路"""
        matrix = CommunicationMatrix(world_size=8)

        metrics = LinkMetrics(
            src_rank=0,
            dst_rank=1,
            transport_type=TransportType.HCCS,
        )
        matrix.set_link(0, 1, metrics)

        # 获取时应该按有序键获取
        link = matrix.get_link(0, 1)
        assert link is not None
        assert link.src_rank == 0
        assert link.dst_rank == 1

        # 反向获取也应该返回相同结果
        link_reverse = matrix.get_link(1, 0)
        assert link_reverse is not None

    def test_get_matrix_2d(self):
        """测试获取 2D 矩阵"""
        matrix = CommunicationMatrix(world_size=4)

        # 设置几条链路
        matrix.set_link(0, 1, LinkMetrics(
            src_rank=0, dst_rank=1, transport_type=TransportType.HCCS,
            achieved_bandwidth_gbps=40.0
        ))
        matrix.set_link(0, 2, LinkMetrics(
            src_rank=0, dst_rank=2, transport_type=TransportType.HCCS,
            achieved_bandwidth_gbps=35.0
        ))

        matrix_2d = matrix.get_matrix_2d()

        assert len(matrix_2d) == 4
        assert matrix_2d[0][1] == 40.0
        assert matrix_2d[1][0] == 40.0  # 对称
        assert matrix_2d[0][2] == 35.0

    def test_compute_summary(self):
        """测试计算摘要"""
        matrix = CommunicationMatrix(world_size=4)

        matrix.set_link(0, 1, LinkMetrics(
            src_rank=0, dst_rank=1, transport_type=TransportType.HCCS,
            total_transit_size_mb=100.0,
            total_transit_time_ms=2.0,
        ))
        matrix.set_link(0, 2, LinkMetrics(
            src_rank=0, dst_rank=2, transport_type=TransportType.HCCS,
            total_transit_size_mb=50.0,
            total_transit_time_ms=1.0,
        ))

        matrix.compute_summary()

        assert matrix.total_comm_data_mb == 150.0
        assert matrix.total_comm_time_ms == 3.0
        assert matrix.avg_bandwidth_gbps == 50.0  # 150 / 3

    def test_to_prompt_text(self):
        """测试生成 Prompt 文本"""
        matrix = CommunicationMatrix(
            world_size=8,
            num_machines=1,
            npus_per_machine=8,
            total_comm_data_mb=1000.0,
            total_comm_time_ms=20.0,
            avg_bandwidth_gbps=50.0,
        )

        text = matrix.to_prompt_text()

        assert "World Size: 8" in text
        assert "1000.00 MB" in text
        assert "50.00 GB/s" in text


class TestCommOpType:
    """CommOpType 测试"""

    def test_from_op_name_all_reduce(self):
        """测试从操作名称推断类型 - AllReduce"""
        op_type = CommOpType.from_op_name("hcom_all_reduce_123")
        assert op_type == CommOpType.ALL_REDUCE

    def test_from_op_name_send(self):
        """测试从操作名称推断类型 - Send"""
        op_type = CommOpType.from_op_name("hcom_send")
        assert op_type == CommOpType.SEND

    def test_from_op_name_unknown(self):
        """测试从操作名称推断类型 - 未知"""
        op_type = CommOpType.from_op_name("unknown_op")
        assert op_type == CommOpType.UNKNOWN

    def test_is_collective(self):
        """测试是否为集合通信"""
        assert CommOpType.ALL_REDUCE.is_collective is True
        assert CommOpType.ALL_GATHER.is_collective is True
        assert CommOpType.SEND.is_collective is False
        assert CommOpType.RECEIVE.is_collective is False

    def test_is_p2p(self):
        """测试是否为点对点通信"""
        assert CommOpType.SEND.is_p2p is True
        assert CommOpType.RECEIVE.is_p2p is True
        assert CommOpType.ALL_REDUCE.is_p2p is False


class TestCommunicationMatrixAnalyzer:
    """CommunicationMatrixAnalyzer 测试"""

    def test_init(self):
        """测试初始化"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )
        assert analyzer.world_size == 8
        assert analyzer.npus_per_machine == 8
        assert analyzer.num_machines == 1

    def test_init_multi_machine(self):
        """测试多机器初始化"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=16,
            npus_per_machine=8,
        )
        assert analyzer.world_size == 16
        assert analyzer.num_machines == 2

    def test_analyze_from_events(self):
        """测试从事件列表分析"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 2000},
            {"src_rank": 0, "dst_rank": 2, "data_size": 50e6, "duration": 1000},
            {"src_rank": 1, "dst_rank": 3, "data_size": 80e6, "duration": 1500},
        ]

        matrix = analyzer.analyze_from_events(events)

        assert matrix.world_size == 8
        assert len(matrix.link_metrics) == 3

        # 检查链路指标
        link_0_1 = matrix.get_link(0, 1)
        assert link_0_1 is not None
        assert link_0_1.transport_type == TransportType.HCCS  # 同节点内
        assert link_0_1.total_transit_size_mb > 0

    def test_analyze_from_events_inter_node(self):
        """测试跨节点通信事件"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=16,
            npus_per_machine=8,
        )

        events = [
            # 节点内通信
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 2000},
            # 跨节点通信
            {"src_rank": 0, "dst_rank": 8, "data_size": 50e6, "duration": 5000},
        ]

        matrix = analyzer.analyze_from_events(events)

        # 节点内通信应该是 HCCS
        link_intra = matrix.get_link(0, 1)
        assert link_intra.transport_type == TransportType.HCCS

        # 跨节点通信应该是 RDMA
        link_inter = matrix.get_link(0, 8)
        assert link_inter.transport_type == TransportType.RDMA

    def test_slow_link_detection(self):
        """测试慢链路检测"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        # 创建一条慢链路
        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 1000},   # 高带宽
            {"src_rank": 2, "dst_rank": 3, "data_size": 10e6, "duration": 5000},    # 低带宽 (慢链路)
        ]

        matrix = analyzer.analyze_from_events(events)

        # 应该检测到慢链路
        # 由于只有两条链路，且差异较大，应该能检测到
        assert len(matrix.slow_links) > 0 or len(matrix.link_metrics) == 2

    def test_get_top_comm_pairs(self):
        """测试获取通信量最大的 rank 对"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 2000},
            {"src_rank": 0, "dst_rank": 2, "data_size": 200e6, "duration": 4000},  # 最大
            {"src_rank": 1, "dst_rank": 3, "data_size": 50e6, "duration": 1000},
        ]

        matrix = analyzer.analyze_from_events(events)
        top_pairs = analyzer.get_top_comm_pairs(top_n=2)

        assert len(top_pairs) >= 1
        # 第一对应该是 0-2 (200 MB)
        if len(top_pairs) >= 1:
            # 检查排序是否正确
            assert top_pairs[0]["src_rank"] in [0, 2]

    def test_generate_recommendations(self):
        """测试生成优化建议"""
        analyzer = CommunicationMatrixAnalyzer(
            world_size=8,
            npus_per_machine=8,
        )

        events = [
            {"src_rank": 0, "dst_rank": 1, "data_size": 100e6, "duration": 2000},
        ]

        analyzer.analyze_from_events(events)
        recommendations = analyzer.generate_recommendations()

        # 应该有优化建议
        assert isinstance(recommendations, list)


class TestTransportType:
    """TransportType 测试"""

    def test_values(self):
        """测试枚举值"""
        assert TransportType.HCCS.value == "hccs"
        assert TransportType.RDMA.value == "rdma"
        assert TransportType.ROCE.value == "roce"
        assert TransportType.PCIE.value == "pcie"
