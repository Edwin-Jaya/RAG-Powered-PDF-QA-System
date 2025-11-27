from groq import Groq
from configs.settings import LLM_API


client = Groq(api_key=LLM_API)

def generate_answer(prompt, context):
    sys_prompt = f"""
    Instructions:
    - Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
    - Utilize the context provided for accurate and specific information.
    - Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
    - If the user query doesnt align with the context, reject with politely!
    Context: {context}
    """
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role":"system",
                "content":f"{sys_prompt}"
            },
            {
                "role":"user",
                "content":f"{prompt}"
            }
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=0.95,
        reasoning_effort="high",
        reasoning_format="hidden",
        stream=True    
    )
    
    text_resp = ""
    
    for chunk in completion:
        text_resp+=chunk.choices[0].delta.content or ""

    return text_resp

