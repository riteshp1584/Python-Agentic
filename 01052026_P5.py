

def generate_prompt(question, context=None):
    if context:
        return f"Context information: {context}\n\nAnswer this question concisely: {question}"
    return f"Answer this question concisely: {question}"

# example use:
prompt_text = generate_prompt("What is the capital of France?")

print(prompt_text)
