import torch

llm_name = 'mistralai/Mistral-7B-Instruct-v0.2'
llm_tokenizer = 'mistralai/Mistral-7B-Instruct-v0.2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ui_port = 7860
