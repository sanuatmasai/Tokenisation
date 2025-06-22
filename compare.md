# Tokenisation Comparison Report

## Original Sentence
> The cat sat on the mat because it was tired.

---

### 🔹 BPE (GPT-2)
- **Tokens:**  
  `['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']`
- **Token IDs:**  
  `[464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032, 13]`
- **Token Count:**  
  `11`

---

### 🔸 WordPiece (BERT)
- **Tokens:**  
  `['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']`
- **Token IDs:**  
  `[101, 1996, 4937, 2938, 2006, 1996, 13523, 2138, 2009, 2001, 5458, 1012, 102]`
- **Token Count:**  
  `11`

---

### ✍️ Notes on Differences
The BPE tokenizer (used by GPT-2) adds a special character `Ġ` before words that follow a space, while WordPiece (used by BERT) breaks text into lowercase subwords with no space markers. WordPiece also wraps input with `[CLS]` and `[SEP]` tokens which aren’t present in BPE. These differences stem from their training objectives: BPE aims to compress frequent word chunks, while WordPiece balances vocabulary size and coverage of unknown words through subword decomposition. This leads to slight differences in token ID sequences even for the same sentence.