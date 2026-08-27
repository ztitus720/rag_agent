from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from pathlib import Path
import shutil

from app.config import settings
from app.rag.vectorstore import search
from app.agent.graph import run_agent
from app.agent.memory import get_history, add_message
from app.llm.client import chat_stream


router = APIRouter()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    root = Path(settings.document_dir)
    root.mkdir(parents=True, exist_ok=True)

    target = root / file.filename

    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "message": "Run python -m app.rag.ingest to index it."
    }


@router.post("/search")
def search_documents(payload: dict):
    q = payload.get("query", "")

    return (
        {
            "results": search(
                q,
                int(payload.get("top_k", 4))
            )
        }
        if q
        else {"results": []}
    )


@router.post("/chat")
def chat_endpoint(payload: dict):
    message = payload.get("message", "")
    session_id = payload.get("session_id", "default")

    if not message:
        return {
            "answer": "Please provide a message."
        }

    history = get_history(session_id)

    result = run_agent(
        message,
        history=history,
    )

    answer = result.get("answer", "")

    add_message(
        session_id,
        "user",
        message,
    )

    add_message(
        session_id,
        "assistant",
        answer,
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "used_rag": result.get("use_rag", False),
        "tool": result.get("tool", "DIRECT"),
        "context": result.get("context", ""),
    }


@router.post("/chat/stream")
def chat_stream_endpoint(payload: dict):
    message = payload.get("message", "")
    session_id = payload.get("session_id", "default")

    if not message:
        return StreamingResponse(
            iter(["Please provide a message."]),
            media_type="text/plain"
        )

    def generate():
        history = get_history(session_id)

        result = run_agent(
            message,
            history=history,
        )

        context = result.get(
            "context",
            "No context retrieved."
        )

        prompt = f"""User question:
{message}

Selected tool:
{result.get("tool", "DIRECT")}

Knowledge base context:
{context}

Tool result:
{result.get("tool_result", "No tool result.")}

Provide the final answer to the user.
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI knowledge assistant. "
                    "Use the provided context and tool results "
                    "to answer accurately. Do not invent facts."
                )
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        full_answer = ""

        for token in chat_stream(messages):
            full_answer += token
            yield token

        add_message(
            session_id,
            "user",
            message,
        )

        add_message(
            session_id,
            "assistant",
            full_answer,
        )

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )