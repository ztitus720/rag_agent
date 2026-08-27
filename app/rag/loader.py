from pypdf import PdfReader
from pathlib import Path 

SUPPORTED={".pdf",".txt",".md"}

def load_file(path: Path):
    if path.suffix.lower()==".pdf":
        reader=PdfReader(str(path))
        return "\n\n".join(
            f"[Page {i}]\n{p.extract_text() or ''}"
            for i,p in enumerate(reader.pages,1)
        )
    if path.suffix.lower() in {".txt",".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path.suffix}")
