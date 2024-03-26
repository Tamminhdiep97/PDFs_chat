import chromadb
from chromadb.config import Settings


class VectorDB(object):
    def __init__(self, config):
        self.config = config
        self.connect(self.config)
        self.create_test_collection()

    def connect(self, config) -> None:
        self.chroma_client = chromadb.HttpClient(
                host='chatbot_PDFs_db',
                port=8000,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )


    def create_collection(self, collection_name):
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

    def del_colletion(self, collection_name):
        self.chroma_client.delete_collection(collection_name)

    def create_test_collection(self):
        self.test_collection = self.chroma_client.get_or_create_collection("test_name")

    def delete_test_collection(self):
        self.chroma_client.delete_collection("test_name")
