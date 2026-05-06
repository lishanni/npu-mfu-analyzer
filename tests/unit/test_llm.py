"""
LLM 模块测试
"""

import pytest


class TestLLMInterface:
    """LLM 接口测试"""
    
    def test_message_to_dict(self):
        """测试消息转换"""
        from npu_mfu_analyzer.llm.llm_interface import Message
        
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        
        assert d["role"] == "user"
        assert d["content"] == "Hello"
    
    def test_llm_config_default(self):
        """测试默认配置"""
        from npu_mfu_analyzer.llm.llm_interface import LLMConfig
        
        config = LLMConfig()
        
        assert config.backend == "openai"
        assert config.temperature == 0.1
        assert config.max_tokens == 4096
    
    def test_llm_factory_create_mock(self):
        """测试创建 Mock 后端"""
        from npu_mfu_analyzer.llm.llm_interface import LLMConfig, LLMFactory, MockBackend
        
        config = LLMConfig(backend="mock")
        llm = LLMFactory.create(config)
        
        assert isinstance(llm, MockBackend)

    def test_claude_default_model_is_glm(self):
        """测试 claude 后端默认模型映射到 GLM-4.7"""
        from src.llm.llm_interface import LLMConfig

        config = LLMConfig(backend="claude")

        assert config.model == "GLM-4.7"
    
    def test_llm_factory_invalid_backend(self):
        """测试无效后端"""
        from npu_mfu_analyzer.llm.llm_interface import LLMConfig, LLMFactory
        
        config = LLMConfig(backend="invalid")
        
        with pytest.raises(ValueError):
            LLMFactory.create(config)
    
    @pytest.mark.asyncio
    async def test_mock_backend_complete(self):
        """测试 Mock 后端响应"""
        from npu_mfu_analyzer.llm.llm_interface import LLMConfig, LLMFactory, Message
        
        config = LLMConfig(backend="mock")
        llm = LLMFactory.create(config)
        
        messages = [Message(role="user", content="Hello")]
        response = await llm.complete(messages)
        
        assert response.content is not None
        assert response.model == "mock"
        assert "total_tokens" in response.usage


class TestPrompts:
    """Prompt 模板测试"""
    
    def test_system_prompts_exist(self):
        """测试系统 Prompt 存在"""
        from npu_mfu_analyzer.llm.prompts import (
            PERFORMANCE_EXPERT_SYSTEM,
            TIMELINE_ANALYSIS_SYSTEM,
            MFU_ANALYSIS_SYSTEM,
        )
        
        assert len(PERFORMANCE_EXPERT_SYSTEM) > 100
        assert "昇腾" in PERFORMANCE_EXPERT_SYSTEM or "NPU" in PERFORMANCE_EXPERT_SYSTEM
        assert len(TIMELINE_ANALYSIS_SYSTEM) > 100
        assert len(MFU_ANALYSIS_SYSTEM) > 100
