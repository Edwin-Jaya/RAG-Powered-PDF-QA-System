from openai import OpenAI
from configs.settings import LM_STUDIO_URL, EMBED_MODEL

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

def embed_texts(text, model=EMBED_MODEL):
    print("Start Embed")
    embedded = []
    embeddings = client.embeddings.create(input=text, model=model).data
    for index in range(0,len(embeddings)):
        embedded.append(embeddings[index].embedding)
    # print(len(embedded))
    return embedded