import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_sbert_mpnet(conf):
    return HuggingFaceEmbeddings(model_name=conf.emb_model, model_kwargs={'device': conf.device})
