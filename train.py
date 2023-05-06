from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from langchain.chat_models import ChatOpenAI
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
documents = SimpleDirectoryReader('data').load_data()
index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist()