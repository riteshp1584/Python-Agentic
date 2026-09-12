import os
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login, hf_hub_download
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

print(f"Hugging Face will save models to: {os.getenv('HF_HOME')}")

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03
    ),
)
