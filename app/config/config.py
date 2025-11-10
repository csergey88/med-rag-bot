import os

HF_TOKEN=os.getenv("HF_TOKEN")

HUGGINGFACEHUB_REPO_ID="google/flan-t5-large"
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="data"
CHUNK_SIZE=1000
CHUNK_OVERLAP=20