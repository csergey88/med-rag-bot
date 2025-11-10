from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
logger=get_logger(__name__)
from app.common.custom_exception import CustomException

from app.config.config import HF_TOKEN
from app.config.config import HUGGINGFACEHUB_REPO_ID


def get_embedding_model():
    try:
        model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device":"cpu"})
        return model
    except Exception as e:
        logger.error(f"Error in get_embedding_model: {str(e)}")
        raise CustomException(f"Error in get_embedding_model: {str(e)}")

