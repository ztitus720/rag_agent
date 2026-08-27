from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.embedding_model,
            local_files_only=True
        )
    return _model

def embed_documents(texts):
    return get_embedding_model().encode(
        texts,
        normalize_embeddings=True
    ).tolist()

def embed_query(text):
    return get_embedding_model().encode(
        [text],
        normalize_embeddings=True
    )[0].tolist()