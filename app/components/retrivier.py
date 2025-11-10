from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger=get_logger(__name__)

from app.components.vector_store import get_vector_store
from app.components.llm import load_llm

from app.config.config import HUGGINGFACEHUB_REPO_ID, HF_TOKEN

import os

CUSTOM_PROMPT_TEMPLATE="""
<system>
You are a helpful assistant that can answer questions about medical documents.
</system>

<context>
{context}
</context>

<question>
{question}
</question>

Answer:
"""

PROMPT_TEMPLATE=PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def create_retrieval_qa_chain():
    try:
        logger.info("Creating retrieval QA chain")
        vector_store=get_vector_store()
        if not vector_store:
            raise CustomException("Vector store not found")
        llm=load_llm()
        if not llm:
            raise CustomException("LLM not found")
        retrieval_qa_chain=RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k":1}),
            return_source_documents=False,
            chain_type_kwargs={
                "prompt":PROMPT_TEMPLATE
            }
        )
        logger.info("Retrieval QA chain created successfully")
        return retrieval_qa_chain
    except Exception as e:
        logger.error(f"Error in create_retrieval_qa_chain: {str(e)}")
        raise CustomException(f"Error in create_retrieval_qa_chain: {str(e)}")
        
        
