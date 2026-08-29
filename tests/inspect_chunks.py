import chromadb
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_collection("knowledge_base")

result = collection.get(
    include=["documents", "metadatas"]
)

documents = result["documents"]
metadatas = result["metadatas"]

print("COUNT:", len(documents))

for i, (doc, meta) in enumerate(zip(documents, metadatas)):
    print(f"\n--- RECORD {i} ---")
    print("META:", meta)
    print("TEXT:", doc[:500])
