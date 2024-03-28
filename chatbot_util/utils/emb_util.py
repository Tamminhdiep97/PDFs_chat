import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class EMB(object):
    def __init__(self, config):
        self.device = torch.device('cpu')
        self.model = self.create_sbert_mpnet(config)

    def create_sbert_mpnet(self, conf):
        return HuggingFaceEmbeddings(model_name=conf.emb_model, model_kwargs={'device': self.device})
