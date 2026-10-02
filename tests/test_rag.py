from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_text_chunking():
    text = "Python " * 500

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_empty_text():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text("")

    assert chunks == []


def test_small_text_stays_one_chunk():
    text = "Python and C++"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text