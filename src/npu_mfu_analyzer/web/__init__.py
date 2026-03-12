"""Web 模块 - FastAPI 后端和 Web UI"""

try:
    from npu_mfu_analyzer.web.app import app, create_app
    __all__ = ["app", "create_app"]
except ImportError:
    # FastAPI 未安装
    app = None
    create_app = None
    __all__ = []
