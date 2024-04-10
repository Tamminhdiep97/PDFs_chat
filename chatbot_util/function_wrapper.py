import os
from os.path import join as opj
import time

from loguru import logger
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.retrievers import WeaviateHybridSearchRetriever

import utils
import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.collection_name = 'mock'
        self.conf = conf
        self.embedding = utils.EMB(self.conf).model
        self.vectorDB_client = utils.VectorDB(self.conf, self.embedding)
        
        self.vectorDb = WeaviateVectorStore(
            client=self.vectorDB_client.client,
            index_name=self.collection_name,
            text_key='text'
        )
        # self.vectorDb = WeaviateHybridSearchRetriever(
        #     client=self.vectorDB_client.client,
        #     index_name=self.collection_name,
        #     text_key="text",
        #     attributes=[],
        #     create_schema_if_missing=True,
        # ) 
        self.vectorDB_client.create_collection(self.collection_name)
        self.llm = utils.LLM(self.conf)
        # self.chroma_db.delete_test_collection()
        # self.vectorDb = Chroma(
        #     client=self.chroma_db.chroma_client,
        #     # collection_name="test_name",
        #     embedding_function=self.embeddingre,
        #     # persist_directory='document_data'
        # )
        # collection = self.vectorDb.get()
        # logger.info('There are {} documents in collection'.format(str(len(collection.get('ids', [])))))
        # self.vector_db_pdf()
        self.reload_retrieval()
        collection = self.vectorDB_client.get_collection(self.collection_name)
        logger.info(collection)
        response = collection.query.fetch_objects(
            limit=3, include_vector=True
        )
        logger.info(response)
        # for item in collection.iterator():
        #     logger.info(item)

    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        pdf_path = self.conf.pdf_path
        logger.info(os.getcwd())
        for item in os.listdir(pdf_path):
            file_path = opj(pdf_path, item)
            self.emb_document(file_path)
            self.reload_retrieval()

    def emb_document(self, path) -> None:
        # load the document
        logger.info(path)
        loader = PDFPlumberLoader(path)
        documents = loader.load()
        # Split the text
        text_splitter = CharacterTextSplitter(
                chunk_size=conf.emb_chunk_size,
                chunk_overlap=conf.emb_chunk_overlap
            )
        texts = text_splitter.split_documents(documents)
        # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        text_splitter = TokenTextSplitter(
                chunk_size=conf.emb_chunk_size,
                chunk_overlap=conf.emb_chunk_overlap,
                encoding_name='cl100k_base'
            )  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts)
        # Upload Documents
        ## Check if the collection exists. If empty, create a new one with the documents and embeddings.
        # collection = self.vectorDB_client.get_collection(self.collection_name)
        # if len(collection.get('ids', [])) == 0:
        #     logger.info('Create new db collection')
        #     self.vectorDb = self.vectorDb.from_documents(
        #             documents=texts,
        #             embedding=self.embedding,
        #             # persist_directory=self.conf.db_persist_directory,
        #             collection_name="test_name"
        #     )
        # else:
        ## If collection already has documents, sumply add the new ones with their embeddings
        logger.info('Add to exists')
        self.vectorDb.add_documents(
            texts,
            embeddings=self.embedding,
            collection=self.collection_name
        )
        collection = self.vectorDB_client.get_collection(self.collection_name)
        logger.info(collection)
        # collection = self.vectorDb.get()
        # logger.info(
        #     'There are {} documents in collection'.format(
        #         str(
        #             len(collection.get('ids', []))
        #         )
        #     )
        # )                 

    def reload_retrieval(self) -> None:
        self.retriever = self.vectorDb.as_retriever(
            collection=self.collection_name,
            search_kwaprgs={'k={}'.format(str(self.conf.search_topk))},
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm.hf_llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def answer_query(self, question:str) -> str:
        """
        Answer the question
        """
        answer_dict = self.qa.invoke({'query': question,})
        answer = answer_dict['result']
        return answer

if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    logger.info('load helper Function done')
    questions = ['what is lora?', 'what is tokenization']
    for i in questions:
        answer = function_runtime.answer_query(i)
        logger.info('Question: {}'.format(i))
        logger.info('Answer: {}'.format(answer))
