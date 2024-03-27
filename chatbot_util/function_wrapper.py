import os
from os.path import join as opj

from loguru import logger
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

import utils
import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.conf = conf
        self.llm = utils.LLM(self.conf)
        self.embedding = utils.EMB(self.conf).model
        self.chroma_db = utils.VectorDB(self.conf)
        self.vectorDb = Chroma(
            client=self.chroma_db.chroma_client,
            collection_name="test_name",
            embedding_function=self.embedding,
        )
        self.vector_db_pdf()

    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        pdf_path = self.conf.pdf_path
        # persist_directory = self.config.get("persist_directory",None)
        # if persist_directory and os.path.exists(persist_directory):
        #     ## Load from the persist db
        #     self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        # elif pdf_path and os.path.exists(pdf_path):
        ## 1. Extract the documents
        documents = []
        logger.info(os.getcwd())
        for item in os.listdir(pdf_path):
            loader = PDFPlumberLoader(opj(pdf_path, item))
            documents.extend(loader.load())
        ## 2. Split the texts
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        texts = text_splitter.split_documents(documents)
        # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=32, encoding_name='cl100k_base')  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts)
        logger.info(texts)
        ## 3. Create Embeddings and add to chroma store
        ##TODO: Validate if self.embedding is not None
        self.data = self.vectorDb.from_documents(documents=texts, embedding=self.embedding)  # , persist_directory=persist_directory)
        self.retriever = self.data.as_retriever(search_kwaprgs={'k=3'})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm.hf_llm, chain_type="stuff", retriever=self.retriever)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True

    def emb_document(self, path):
        # load the document
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
        collection = self.vectorDb.get()
        if len(collection.get('ids', [])) == 0:
            # self.data = Chroma.from_documents(
            #         documents=texts,
            #         embedding=self.embedding,
            #         persist_directory=self.conf.db_persist_directory,
            #         collection_name="test_name"
            # )
            self.data = self.vectorDb.from_documents(
                    documents=texts,
                    embedding=self.embedding,
                    persist_directory=self.conf.db_persist_directory,
                    collection_name="test_name"
            )
        else:
        ## If collection already has documents, sumply add the new ones with their embeddings
            collection.append(texts, embeddings=self.embedding)
            self.vectorDb.persist()  # Save the updated collection



        

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
