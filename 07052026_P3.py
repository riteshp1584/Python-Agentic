from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate

# 1. Configuration for High-Precision Financial Logic
# Use deepseek-r1:1.5b for math/reasoning tasks
llm = ChatOllama(
    model="deepseek-r1:1.5b",

    # --- PARAMETER TUNING ---
    temperature=0.1,  # Low temperature (0.1-0.3) is best for math/finance to avoid "hallucinations"
    top_p=0.9,  # Focuses on the most likely 90% of tokens; keeps the output coherent
    # top_k=40,         # (Optional) limits the vocabulary to the top 40 words; top_p is usually sufficient

    num_predict=2048,  # This is the "Maximum Tokens" parameter in Ollama

    reasoning=True,    # Only keep this TRUE for DeepSeek-R1 models
    keep_alive='5s'
)

# 2. Execution
topic = "impact of a 50bps rate hike on a 10-year bond price"

# Call the LLM directly with your formatted string
response = llm.invoke(f"Calculate the {topic}")

print(response.content)
