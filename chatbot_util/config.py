import os
from os.path import join as opj

import torch

llm_name = 'mistralai/Mistral-7B-Instruct-v0.2'
llm_tokenizer = 'mistralai/Mistral-7B-Instruct-v0.2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ui_port = 7860

pdf_path = opj('chatbot_util', 'media') # None is for Online Upload file mode

emb_model = 'sentence-transformers/all-mpnet-base-v2'
emb_chunk_size = 512
emb_chunk_overlap = 32

search_topk = 3

db_persist_directory = 'document_data'
