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

from . import utils
from . import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.collection_name = 'vision_technique'
        self.conf = conf
        self.embedding = utils.EMB(self.conf).model
        self.vectorDB_client = utils.VectorDB(self.conf, self.embedding)
        # self.vectorDB_client.delete_collection(self.collection_name)
        self.vectorDb = WeaviateVectorStore(
            client=self.vectorDB_client.client,
            index_name=self.collection_name,
            text_key='text',
            embedding=self.embedding
        )

        # self.vectorDB_client.create_collection(self.collection_name)
        self.llm = utils.LLM(self.conf)
        # self.vector_db_pdf()
        self.reload_retrieval()


    # def vector_db_pdf(self) -> None:
    #     """
    #     creates vector db for the embeddings and persists them or loads a vector db from the persist directory
    #     """
    #     pdf_path = opj(self.conf.pdf_path, self.collection_name)
    #     logger.info(os.getcwd())
    #     for item in os.listdir(pdf_path):
    #         file_path = opj(pdf_path, item)
    #         self.emb_document(file_path)
            # self.reload_retrieval()

    def emb_document(self, path, user) -> None:
        # load the document
        logger.info(path)
        loader = PDFPlumberLoader(path)
        documents = loader.load()
        for document in documents:
            document.metadata = {'user': user}
        logger.info(documents[0].metadata)
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

        logger.info('Add to exists')
        self.vectorDb.add_documents(
            texts,
            embeddings=self.embedding,
            collection=self.collection_name
        )

    def reload_retrieval(self, collection_name=None) -> None:
        if collection_name is not None:
            self.collection_name = collection_name
        logger.info(self.collection_name)
        self.vectorDb = WeaviateVectorStore(
            client=self.vectorDB_client.client,
            index_name=self.collection_name,
            text_key='text',
            embedding=self.embedding
        )

        self.retriever = self.vectorDb.as_retriever(
            collection=self.collection_name,
            search_kwaprgs={'k={}'.format(str(self.conf.search_topk))},
        )
        # self.retriever = self.collection.as_triever()
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm.hf_llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        logger.info(self.qa)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def answer_query(self, question:str) -> str:
        """
        Answer the question
        """
        answer_dict = self.qa.invoke({'query': question,})
        logger.info(answer_dict['result'])
        # logger.info(type(answer_dict['result']))
        answer = answer_dict['result'].split('Helpful Answer: ')[1]
        return answer

if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    logger.info('load helper Function done')
    questions = ['what is lora?', 'what is tokenization']
    for i in questions:
        answer = function_runtime.answer_query(i)
        logger.info('Question: {}'.format(i))
        logger.info('Answer: {}'.format(answer))
