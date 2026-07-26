# runner 子包：AgentRunner 的私有实现。
#
# 只暴露 AgentRunner 一个名字。所有子模块（_agent、_hooks、
# _internals、_factory）都是实现细节，外部不应直接 import。
"""AgentRunner — orchestrates load → build → run → save."""
from agent_framework.runner._agent import AgentRunner

__all__ = ["AgentRunner"]
