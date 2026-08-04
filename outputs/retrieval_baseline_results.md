# Retrieval Baseline Results

| Method | Token F1 | ROUGE-L | BERTScore | Ms/Query | Total Time (s) | Fallback Rate | Reuse Rate | Ret. Mean Chars | Act. Mean Chars | Mismatch? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Match** | 0.0495 | 0.0390 | 0.6452 | 0.41ms | 4.42s | 100.00% | 100.00% | 67.0 | 632.8 | ⚠️ YES |
| **TF-IDF** | 0.1834 | 0.1147 | 0.6977 | 5.24ms | 57.00s | 0.00% | 22.84% | 633.9 | 632.8 | ✅ NO |
| **BM25** | 0.1831 | 0.1140 | 0.6981 | 5.55ms | 60.38s | 0.00% | 20.17% | 636.3 | 632.8 | ✅ NO |
| **Sentence Embedding** | 0.1627 | 0.1002 | 0.6892 | 65.34ms | 710.85s | 0.00% | 24.77% | 633.3 | 632.8 | ✅ NO |

## Baseline Method Comparison
- **Exact Match Retrieval**: Evaluates direct prompt reuse. When prompts don't match exactly, fallback text is returned.
- **TF-IDF Retrieval**: Character & word TF-IDF cosine similarity nearest neighbor retrieval.
- **BM25 Retrieval**: Okapi BM25 ranking over tokenized Bengali text.
- **Sentence-Embedding Retrieval**: Multilingual dense embedding (`paraphrase-multilingual-MiniLM-L12-v2`) similarity retrieval.