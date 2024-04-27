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
max_new_token = 2048

search_topk = 3

db_persist_directory = 'document_data'


APP_NAME = 'PDFs_chatbot'

API_VERSION = '0.0.1'
API_DESCRIPTION = 'PDFs Chatbot Backend API'
API_PREFIX = '/api'

LOG_SYSTEM = 'SYSTEM'
LOG_LEVEL = 'INFO'
LOG_LEVEL_SYSTEM = 100
if LOG_LEVEL == 'DEBUG':
    DEBUG = True
else:
    DEBUG = False
ROTATION = '500 MB'
RETENTION = '10 days'
