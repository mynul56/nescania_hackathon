# Data Audit

## Source Files
- train: `train.csv`
- test: `test.csv`
- sample submission present: `False`

## Inferred Columns
- prompt column: `input`
- response column: `output`

## Schema
- train rows: `108954`
- test rows: `1000`
- train columns: `id, input, output`
- test columns: `id, input`

## Duplicate Findings
- train full-row duplicates: `0`
- test full-row duplicates: `0`
- train prompt duplicates: `205`
- test prompt duplicates: `0`
- exact cross-split prompt overlap: `0`
- near-duplicate method: `hashing_char_wb_embeddings`
- near-duplicate threshold: `0.9`
- near cross-split pairs found: `0`
- conflicting prompt groups in train: `50`

## Split Recommendation
- strategy: Fixed seed, content-aware split by prompt text with exact and near-duplicate groups kept together, length-bucket stratification, and category stratification if a category field exists.
- seed: `42`
- grouping key: `input`
- leakage guard: Re-check exact and embedding-near duplicates across the final train/validation boundary before freezing the split.

## Retrieval Suitability
- assessment: High if responses are repetitive or templated; exact-match and BM25 should be strong baselines, with embedding retrieval useful for paraphrases and lexical drift.
- duplicate prompt groups: `50`
- conflicting prompt groups: `50`
- near-duplicate cross-split pairs: `0`

## Safety Surface
- prompt keyword hits: `{'ব্যথা': 44118, 'বিষ': 13226, 'জ্বর': 6255, 'গর্ভবতী': 5059, 'জরুরি': 4454, 'শ্বাসকষ্ট': 3769, 'খিঁচুনি': 3083, 'রক্তপাত': 2948, 'অজ্ঞান': 1083, 'হার্ট অ্যাটাক': 969, 'স্ট্রোক': 788, 'আত্মহত্যা': 363, 'বুক ব্যথা': 20, 'emergency': 18, 'pain': 8, 'bleeding': 3, 'poison': 3, 'pregnancy': 1, 'stroke': 1}`
- response keyword hits: `{'ব্যথা': 30139, 'বিষ': 26076, 'জ্বর': 5363, 'জরুরি': 4917, 'রক্তপাত': 3363, 'শ্বাসকষ্ট': 3111, 'খিঁচুনি': 2922, 'গর্ভবতী': 2285, 'স্ট্রোক': 826, 'হার্ট অ্যাটাক': 548, 'অজ্ঞান': 495, 'আত্মহত্যা': 175, 'বুক ব্যথা': 52, 'pain': 20, 'pregnancy': 7, 'emergency': 2, 'bleeding': 2, 'stroke': 1}`
- test prompt keyword hits: `{'ব্যথা': 429, 'বিষ': 118, 'জ্বর': 52, 'গর্ভবতী': 47, 'জরুরি': 46, 'শ্বাসকষ্ট': 27, 'খিঁচুনি': 26, 'রক্তপাত': 24, 'স্ট্রোক': 11, 'অজ্ঞান': 8, 'হার্ট অ্যাটাক': 7, 'আত্মহত্যা': 1, 'বুক ব্যথা': 1, 'emergency': 1}`

## Notes
- category-like columns: `[]`
- category note: No obvious category/specialty field detected
