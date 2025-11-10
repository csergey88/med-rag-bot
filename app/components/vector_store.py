from langchain_community.vectorstores import FAISS

from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH
import os

logger=get_logger(__name__)

def get_vector_store():
    try:
        embedding_model=get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            vector_store=FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True 
            )
            return vector_store
        else:
            logger.info("Vector store not found. Creating a new one.")
            return None
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        raise CustomException(f"Error in get_vector_store: {str(e)}")

def save_vector_store(chunks):
    try:
        embedding_model=get_embedding_model()
        vector_store=FAISS.from_documents(chunks,embedding_model)
        vector_store.save_local(DB_FAISS_PATH)
    except Exception as e:
        logger.error(f"Error in save_vector_store: {str(e)}")
        raise CustomException(f"Error in save_vector_store: {str(e)}")