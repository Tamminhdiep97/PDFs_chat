import uuid

import weaviate
import weaviate.classes.config as wvcc
from weaviate.connect import ConnectionParams
from loguru import logger


class VectorDB(object):
    def __init__(self, config, embedding):
        self.config = config
        self.embedding_function = embedding
        self.connect()

    def mock_data(self):
        id_mock = str(uuid.uuid4())
        self.user_data = {
            "user_id": id_mock
        }
        self.session_data = {
            "session_id": "session_1",
            "user": {"$$reference": "User.user_id={}".format(id_mock)}
        }

    
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
        # self.client = weaviate.Client('http://document_db:8080')
        pass

    def create_collection(self, collection_name):
        if not self.client.collections.exists(collection_name):
            self.client.collections.create(
                name=collection_name,
                # vectorizer_config=wvcc.Configure.Vectorizer.text2vec_cohere(),
                vectorizer_config=None,
                generative_config=None,
            #     properties=[
            #         wvcc.Property(
            #             name="Title", data_type=wvcc.DataType.TEXT,
            #             name="Content", data_type=wvcc.DataType.TEXT
            #         )
            #     ]
            )
        pass

    def get_collection(self, collection_name):
        collection = self.client.collections.get(collection_name)
        # for item in collection.iterator(
        #     include_vector=True  # If using named vectors, you can specify ones to include e.g. ['title', 'body'], or True to include all
        # ):
            # logger.info(item.properties)
            # logger.info(item.vector)
        return collection

    def delete_collection(self, collection_name):
        self.client.collections.delete(collection_name)
        pass

    def delete_item(self, collection_name, meta_data):
        pass

    def query(self):
        pass
