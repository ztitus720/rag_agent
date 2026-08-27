from typing import TypedDict, List

class AgentState(TypedDict):
    message: str
    context: str
    answer: str
    use_rag: bool
    tool: str
    tool_result: str
    history: List