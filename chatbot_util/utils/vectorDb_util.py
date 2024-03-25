import chromadb
from chromadb.config import Settings


class VectorDB(object):
    def __init__(self, config):
        self.connect(config)

    def connect(self, config) -> None:
        self.chroma_client = chromadb.HttpClient(
                host='chatbot_PDFs_db',
                port=8000,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
