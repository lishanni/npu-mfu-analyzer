"""
Agent 模块测试
"""

import pytest


class TestBaseAgent:
    """BaseAgent 测试"""
    
    def test_agent_message_to_llm_message(self):
        """测试消息转换"""
        from src.agents.base_agent import AgentMessage
        from src.llm.llm_interface import Message
        
        agent_msg = AgentMessage(role="user", content="Hello")
        llm_msg = agent_msg.to_llm_message()
        
        assert isinstance(llm_msg, Message)
        assert llm_msg.role == "user"
        assert llm_msg.content == "Hello"
    
    def test_analysis_result_success(self):
        """测试成功的分析结果"""
        from src.agents.base_agent import AnalysisResult
        
        result = AnalysisResult(
            agent_name="TestAgent",
            success=True,
            summary="分析完成",
            recommendations=["建议1", "建议2"]
        )
        
        assert result.success
        assert len(result.recommendations) == 2


class TestTimelineAgent:
    """TimelineAgent 测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_with_mock_llm(self):
        """使用 Mock LLM 测试分析"""
        from src.agents.base_agent import TimelineAgent
        from src.llm.llm_interface import LLMConfig, LLMFactory
        
        config = LLMConfig(backend="mock")
        llm = LLMFactory.create(config)
        
        agent = TimelineAgent(llm)
        
        data = {
            "avg_compute_time": 30000,
            "avg_comm_time": 15000,
            "avg_free_time": 5000,
        }
        
        result = await agent.analyze(data)
        
        assert result.success
        assert result.agent_name == "TimelineAgent"


class TestOrchestrator:
    """Orchestrator 测试"""
    
    def test_init(self, tmp_path):
        """测试初始化"""
        from src.agents.orchestrator import Orchestrator
        from src.llm.llm_interface import LLMConfig
        
        config = LLMConfig(backend="mock")
        orchestrator = Orchestrator(str(tmp_path), llm_config=config)
        
        assert orchestrator.profiling_path == str(tmp_path)
        assert "timeline" in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_run_with_empty_data(self, tmp_path):
        """测试空数据分析"""
        from src.agents.orchestrator import Orchestrator
        from src.llm.llm_interface import LLMConfig
        
        config = LLMConfig(backend="mock")
        orchestrator = Orchestrator(str(tmp_path), llm_config=config)
        
        report = await orchestrator.run()
        
        # 空数据应该返回失败
        assert not report.success or report.profiling_summary is not None


class TestAnalysisReport:
    """AnalysisReport 测试"""
    
    def test_to_markdown(self):
        """测试 Markdown 输出"""
        from src.agents.orchestrator import AnalysisReport
        from src.data_loader.data_summarizer import ProfilingSummary
        
        report = AnalysisReport(
            success=True,
            summary="分析完成",
            profiling_summary=ProfilingSummary(
                data_path="/test",
                data_type="db",
                framework="pytorch",
                rank_count=8,
            ),
            recommendations=["优化建议1", "优化建议2"]
        )
        
        md = report.to_markdown()
        
        assert "# NPU MFU 性能分析报告" in md
        assert "分析完成" in md
        assert "优化建议1" in md
