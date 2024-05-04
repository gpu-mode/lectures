from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("bert-base-cased")

print(dir(model))