from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.tools import (
    rag_search,
    format_context,
    calculator,
    web_search,
)
from app.llm.client import chat


ROUTER_PROMPT = '''You are a routing classifier.

Decide which action should be used to answer the user's question.

Return exactly one word:

RAG
CALCULATOR
WEB_SEARCH
DIRECT

Rules:

- Return RAG if the question requires information from the private knowledge base.
- Return CALCULATOR if the question requires arithmetic calculation.
- Return WEB_SEARCH if the question requires current, recent, public, or internet-based information.
- Return DIRECT for general conversation or questions that do not require the private knowledge base, calculation, or web search.

Question:
{question}
'''


SYSTEM_PROMPT = '''You are a helpful AI knowledge assistant.

Answer the user's question using the information provided by the selected tool.

For RAG questions:
- Use ONLY the provided knowledge base context.
- Do not add facts that are not explicitly supported by the context.
- Do not infer unstated relationships.
- If the context is insufficient, clearly say that the information is not available.

For calculator questions:
- Use the calculator result provided by the tool.
- Do not change the calculated result.

For web search questions:
- Use the web search results as the primary source of current public information.
- Do not invent information that is not supported by the search results.
- Mention relevant sources or URLs when appropriate.

For general questions:
- Answer normally and helpfully.
'''


def route_node(state):
    result = chat(
        [
            {
                "role": "system",
                "content": ROUTER_PROMPT.format(
                    question=state["message"]
                ),
            }
        ]
    ).strip().upper()

    if result.startswith("RAG"):
        tool = "RAG"

    elif result.startswith("CALCULATOR"):
        tool = "CALCULATOR"

    elif result.startswith("WEB_SEARCH"):
        tool = "WEB_SEARCH"

    else:
        tool = "DIRECT"

    return {
        "tool": tool,
        "use_rag": tool == "RAG",
    }


def retrieve_node(state):
    results = rag_search(
        state["message"],
        4,
    )

    return {
        "context": format_context(results)
    }


def calculator_node(state):
    result = calculator(
        state["message"]
    )

    return {
        "tool_result": result
    }


def web_search_node(state):
    result = web_search(
        state["message"],
        5,
    )

    return {
        "tool_result": result
    }


def answer_node(state):
    history = state.get("history", [])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": f'''User question:
{state["message"]}

Selected tool:
{state.get("tool", "DIRECT")}

Knowledge base context:
{state.get("context", "No context retrieved.")}

Tool result:
{state.get("tool_result", "No tool result.")}

Provide the final answer to the user.
''',
        }
    )

    answer = chat(messages)

    return {
        "answer": answer
    }


def choose_path(state):
    return state["tool"].lower()


def build_graph():
    g = StateGraph(AgentState)

    g.add_node(
        "route",
        route_node,
    )

    g.add_node(
        "retrieve",
        retrieve_node,
    )

    g.add_node(
        "calculator",
        calculator_node,
    )

    g.add_node(
        "web_search",
        web_search_node,
    )

    g.add_node(
        "answer",
        answer_node,
    )

    g.set_entry_point("route")

    g.add_conditional_edges(
        "route",
        choose_path,
        {
            "rag": "retrieve",
            "calculator": "calculator",
            "web_search": "web_search",
            "direct": "answer",
        },
    )

    g.add_edge(
        "retrieve",
        "answer",
    )

    g.add_edge(
        "calculator",
        "answer",
    )

    g.add_edge(
        "web_search",
        "answer",
    )

    g.add_edge(
        "answer",
        END,
    )

    return g.compile()


agent = build_graph()

def run_agent(message, history=None):
    if history is None:
        history = []

    return agent.invoke(
        {
            "message": message,
            "context": "",
            "answer": "",
            "use_rag": False,
            "tool": "DIRECT",
            "tool_result": "",
            "history": history,
        }
    )