import os
from dotenv import load_dotenv
from tavily import TavilyClient
import ast
import operator

from app.rag.vectorstore import search


def rag_search(query, top_k=4):
    return search(query, top_k)


def format_context(results):
    if not results:
        return "No relevant documents were found."

    return "\n\n".join(
        f"[Source {i}: {x['metadata'].get('source', 'unknown')}, "
        f"chunk {x['metadata'].get('chunk_index', '?')}]\n{x['text']}"
        for i, x in enumerate(results, 1)
    )


# --------------------------------------------------
# Calculator Tool
# --------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.UnaryOp):
        operator_func = _ALLOWED_OPERATORS.get(type(node.op))

        if operator_func is None:
            raise ValueError("Unsupported unary operator.")

        return operator_func(
            _safe_eval(node.operand)
        )

    if isinstance(node, ast.BinOp):
        operator_func = _ALLOWED_OPERATORS.get(type(node.op))

        if operator_func is None:
            raise ValueError("Unsupported operator.")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        return operator_func(left, right)

    raise ValueError("Unsupported expression.")


def calculator(expression: str):
    """
    Safely evaluate basic arithmetic expressions.

    Supported:
    +, -, *, /, %, **, parentheses
    """

    try:
        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = _safe_eval(tree.body)

        return str(result)

    except Exception as e:
        return f"Calculator error: {e}"

# --------------------------------------------------
# Web Search Tool
# --------------------------------------------------

load_dotenv()

_tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def web_search(query: str, max_results: int = 5):
    """
    Search the web using Tavily.
    """

    try:
        response = _tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )

        results = response.get("results", [])

        if not results:
            return "No web search results were found."

        formatted = []

        for i, result in enumerate(results, 1):
            formatted.append(
                f"[Web Result {i}]\n"
                f"Title: {result.get('title', 'Unknown')}\n"
                f"URL: {result.get('url', 'Unknown')}\n"
                f"Content: {result.get('content', '')}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {e}"