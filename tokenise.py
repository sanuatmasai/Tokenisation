from transformers import AutoTokenizer, pipeline
import json

sentence = "The cat sat on the mat because it was tired."
masked_sentence = "The cat sat on the [MASK] because it was [MASK]."

# --- Tokenization ---

# 1. BPE (GPT-2)
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bpe_tokens = bpe_tokenizer.tokenize(sentence)
bpe_ids = bpe_tokenizer.encode(sentence)

# 2. WordPiece (BERT)
wp_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
wp_tokens = wp_tokenizer.tokenize(sentence)
wp_ids = wp_tokenizer.encode(sentence)

# --- Fill-in-the-Blank Predictions ---

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

first_mask = "The cat sat on the [MASK] because it was tired."
first_mask_predictions = fill_mask(first_mask)[:3]

top_first_token = first_mask_predictions[0]['token_str']
second_mask = f"The cat sat on the {top_first_token} because it was [MASK]."
second_mask_predictions = fill_mask(second_mask)[:3]

# Save results
output = {
    "tokenisation": {
        "sentence": sentence,
        "bpe": {
            "tokens": bpe_tokens,
            "token_ids": bpe_ids,
            "token_count": len(bpe_tokens)
        },
        "wordpiece": {
            "tokens": wp_tokens,
            "token_ids": wp_ids,
            "token_count": len(wp_tokens)
        }
    },
    "mask_predictions": {
        "masked_sentence": masked_sentence,
        "first_mask_top_3": [
            {"token": r["token_str"], "score": round(r["score"], 4)} for r in first_mask_predictions
        ],
        "second_mask_top_3": [
            {"token": r["token_str"], "score": round(r["score"], 4)} for r in second_mask_predictions
        ]
    }
}

with open("predictions.json", "w") as f:
    json.dump(output["mask_predictions"], f, indent=2)

# Print output for report generation
print("\n=== BPE Tokens ===")
print(bpe_tokens)
print("Token IDs:", bpe_ids)
print("Token Count:", len(bpe_tokens))

print("\n=== WordPiece Tokens ===")
print(wp_tokens)
print("Token IDs:", wp_ids)
print("Token Count:", len(wp_tokens))

print("\n✅ Predictions saved to predictions.json")
