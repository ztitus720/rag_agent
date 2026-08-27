from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Intelligent Knowledge Assistant", version="0.1.0")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-agent"}
