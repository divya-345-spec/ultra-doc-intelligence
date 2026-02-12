def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

            start = end - overlap
            if start < 0:
                start = 0

    return chunks
