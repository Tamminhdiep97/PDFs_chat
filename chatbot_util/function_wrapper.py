import os

from loguru import logger
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.chains import RetrievalQA

import utils
import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.llm = utils.LLM(conf)
        self.emb = utils.EMB(conf)
        self.vectorDb = utils.VectorDB(conf)
        self.retriever = self.vectorDb.chroma_client.as_retriver(search_kwargs={'k=3'})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm.hf_llm, chain_type="stuff", retriever=self.retriever)

    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        pdf_path = self.config.get("pdf_path",None)
        persist_directory = self.config.get("persist_directory",None)
        if persist_directory and os.path.exists(persist_directory):
            ## Load from the persist db
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        elif pdf_path and os.path.exists(pdf_path):
            ## 1. Extract the documents
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            ## 2. Split the texts
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
            text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = self.vectorDb.chroma_client.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        else:
            raise ValueError("NO PDF found")


if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    logger.info('load 3 model done')
