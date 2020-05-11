from transformers import pipeline

fill_mask = pipeline(
	"fill-mask",
	model="outputs/generator_model",
	tokenizer="outputs/generator_model"
)

print(
	fill_mask(f"HuggingFace is creating a {fill_mask.tokenizer.mask_token} that the community uses to solve NLP tasks.")
)