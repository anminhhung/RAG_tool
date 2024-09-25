def split_text(text, max_tokens) -> list[str]:
    words = text.split()  # Tokenize by spaces and newlines
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_count = len(word.split())  # Number of tokens in the word

        # If adding the current word exceeds max_tokens, finalize the current chunk
        if current_token_count + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]  # Start a new chunk with the current word
            current_token_count = token_count
        else:
            current_chunk.append(word)
            current_token_count += token_count

    # Add the final chunk if any tokens are left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
