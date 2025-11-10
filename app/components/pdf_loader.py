import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
logger=get_logger(__name__)
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH,CHUNK_SIZE,CHUNK_OVERLAP


def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exist")
        
        loader=DirectoryLoader(DATA_PATH,glob="*.pdf",loader_cls=PyPDFLoader)
        documents=loader.load()
        if not documents:
            logger.error("No PDF files found in the data path")
        else:
            logger.info(f"Loaded {len(documents)} PDF files")
        
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {str(e)}")
        #raise CustomException(f"Error loading PDF files: {str(e)}")
        return []

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents found")
        
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        text_chunks=text_splitter.split_documents(documents)
        return text_chunks
    except Exception as e:
        logger.error(f"Error creating text chunks: {str(e)}")
        #raise CustomException(f"Error creating text chunks: {str(e)}")
        return []


