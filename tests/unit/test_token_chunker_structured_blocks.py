from src.core.chunking.token_chunker import TokenChunker


def test_markdown_table_with_decimals_stays_atomic_within_structural_limit():
    table = "\n".join(
        [
            "| Model | BLEU | ROUGE-L | CIDEr |",
            "| --- | --- | --- | --- |",
            "| GPT-2 M | 68. 2 | 71. 0 | 2. 47 |",
            "| GPT-2 L | 70. 4 | 72. 0 | 2. 53 |",
            "| LoRA | 70. 4±. 1 | 72. 0±. 2 | 2. 47±. 02 |",
        ]
    )

    chunks = TokenChunker(max_tokens=80).chunk_text(table)

    assert len(chunks) == 1
    assert chunks[0]["main_content"] == table


def test_oversized_structured_block_splits_by_lines_not_decimal_sentences():
    block = "\n".join(
        ["Figure 8: normalized subspace similarity"]
        + [f"{idx} 0. {idx % 10}5 61. 95 ||Wq||F = 6. 91" for idx in range(35)]
    )

    chunks = TokenChunker(max_tokens=35).chunk_text(block)
    joined = "\n".join(chunk["main_content"] for chunk in chunks)

    assert len(chunks) > 1
    assert joined == block
    assert all(not chunk["main_content"].rstrip().endswith("0.") for chunk in chunks)
