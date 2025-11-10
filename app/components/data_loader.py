import os
from app.components.pdf_loader import load_pdf_files,create_text_chunks
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger=get_logger(__name__)

def process_and_store_pdfs():
    try:
        documents=load_pdf_files()
        text_chunks=create_text_chunks(documents)
        save_vector_store(text_chunks)
        logger.info("Data loaded successfully")
        return text_chunks
    except Exception as e:
        logger.error(f"Error in process_and_store_pdfs: {str(e)}")
        raise CustomException(f"Error in process_and_store_pdfs: {str(e)}") 

if __name__=="__main__":
    process_and_store_pdfs()
            