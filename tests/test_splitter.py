from app.rag.splitter import split_text

def test_split_text():
    chunks=split_text("a"*2000,500,50)
    assert len(chunks)>1
    assert all(len(x)<=500 for x in chunks)
