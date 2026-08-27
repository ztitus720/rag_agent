from pathlib import Path
from app.config import settings
from app.rag.loader import load_file, SUPPORTED
from app.rag.splitter import split_text
from app.rag.vectorstore import add_documents

def ingest():
    root=Path(settings.document_dir)
    root.mkdir(parents=True,exist_ok=True)
    texts=[]; metas=[]
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            chunks=split_text(load_file(path))
            for i,chunk in enumerate(chunks):
                texts.append(chunk)
                metas.append({"source":path.name,"chunk_index":i})
    print(f"Ingested {add_documents(texts,metas)} chunks.")

if __name__=="__main__":
    ingest()
