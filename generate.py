from transformers import ElectraTokenizer, ElectraForMaskedLM
import torch

tokenizer = ElectraTokenizer.from_pretrained('outputs')
model = ElectraForMaskedLM.from_pretrained('outputs/generator_model')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0) # Batch size 1
outputs = model(input_ids, masked_lm_labels=input_ids)

loss, prediction_scores = outputs[:2]

print(loss)