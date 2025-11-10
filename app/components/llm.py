from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.common.logger import get_logger
logger=get_logger(__name__)
from app.common.custom_exception import CustomException
from app.config.config import HF_TOKEN
from app.config.config import HUGGINGFACEHUB_REPO_ID

def load_llm(huggingfacehub_repo_id: str = HUGGINGFACEHUB_REPO_ID, huggingfacehub_api_key: str = HF_TOKEN):
    try:
        logger.info(f"Loading model {huggingfacehub_repo_id} locally...")
        
        # Load tokenizer and model locally
        tokenizer = AutoTokenizer.from_pretrained(huggingfacehub_repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(huggingfacehub_repo_id)
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
        )
        
        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Error in load_llm: {str(e)}")
        raise CustomException(f"Error in load_llm: {str(e)}")
