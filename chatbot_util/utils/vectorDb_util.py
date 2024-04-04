import chromadb
from chromadb.config import Settings
import weaviate
from weaviate.connect import ConnectionParams


class VectorChromaDB(object):
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
        self.create_test_collection()


class VectorDB(object):
    def __init__(self, config, embedding):
        self.config = config
        self.embedding_function = embedding
        self.connect()
    
    def connect(self):
        self.client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host="document_db",
                http_port="8080",
                http_secure=False,
                grpc_host="document_db",
                grpc_port="50051",
                grpc_secure=False,
            ),
            # auth_client_secret=weaviate.auth.AuthApiKey("secr3tk3y"),

            additional_config=weaviate.config.AdditionalConfig(
                startup_period=10,
                timeout=(5, 15)  # Values in seconds
            ),
        )

        self.client.connect()
        # self.client = weaviate.connect_to_local(
        #     port=8080,
        #     grpc_port=50051,
        #     additional_config=weaviate.config.AdditionalConfig(timeout=(5, 15))  # Values in seconds
        # )

        pass

    def create_collection(self):
        pass

    def delete_collection(self):
        pass

    def query(self):
        pass
