from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
nlp = pipeline('sentiment-analysis')
print(
	nlp('We are very happy to include pipeline into the transformers repository.')
)