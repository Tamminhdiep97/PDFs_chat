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
            # documents = loader.load()
            ## 2. Split the texts
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=16)
        texts = text_splitter.split_documents(documents)
        # text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=16)  # This the encoding for text-embedding-ada-002
        texts = text_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
        self.data = self.vectorDb.from_documents(documents=texts, embedding=self.embedding)  # , persist_directory=persist_directory)
        self.retriever = self.data.as_retriever(search_kwaprgs={'k=3'})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm.hf_llm, chain_type="stuff", retriever=self.retriever)
        self.qa.combine_documents_chain.verbose = True
        self.qa.return_source_documents = True
        # self.vectordb.persist()
        # else:
        #     raise ValueError("NO PDF found")

    # def retreival_qa_chain(self):
    #     """
    #     Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
    #     """
    #     ##TODO: Use custom prompt
    #     # self.retriever = self.vectordb.as_retriever(search_kwargs={"k":3})
    #     # 
    #     # if self.config["llm"] == LLM_OPENAI_GPT35:
    #     #   # Use ChatGPT API
    #     #   self.qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.), chain_type="stuff",\
    #     #                               retriever=self.vectordb.as_retriever(search_kwargs={"k":3}))
    #     # else:
    #     #     hf_llm = HuggingFacePipeline(pipeline=self.llm,model_id=self.config["llm"])

    #     self.qa = RetrievalQA.from_chain_type(llm=shf_llm, chain_type="stuff",retriever=self.retriever)
    #     # if self.config["llm"] == LLM_FLAN_T5_SMALL or self.config["llm"] == LLM_FLAN_T5_BASE or self.config["llm"] == LLM_FLAN_T5_LARGE:
    #     #     question_t5_template = """
    #     #     context: {context}
    #     #     question: {question}
    #     #     answer: 
    #     #     """
    #     #     QUESTION_T5_PROMPT = PromptTemplate(
    #     #         template=question_t5_template, input_variables=["context", "question"]
    #     #     )
    #     #     self.qa.combine_documents_chain.llm_chain.prompt = QUESTION_T5_PROMPT
    #     self.qa.combine_documents_chain.verbose = True
    #     self.qa.return_source_documents = True
    def answer_query(self, question:str) ->str:
        """
        Answer the question
        """

        answer_dict = self.qa({'query': question,})
        answer = answer_dict['result']
        return answer

if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    logger.info('load helper Function done')
    questions = ['what is lora?', 'how can i train/finetune a llm on a new language?']
    for i in questions:
        logger.info(i)
        answer = function_runtime.answer_query(i)
        logger.info(answer)
