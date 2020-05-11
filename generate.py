from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model = AutoModelWithLMHead.from_pretrained("outputs/generator_model"),
    tokenizer= AutoTokenizer.from_pretrained("outputs/generator_model")
)

results = fill_mask(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks.")
print(results)