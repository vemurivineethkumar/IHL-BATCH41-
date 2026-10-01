

from openai import OpenAI

# OpenAI-compatible LLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

MODEL = "your-model-name"

conversation_history = []

def chat(user_message):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer


print("=== USE CASE 1: BASELINE LLM CHATBOT ===")

while True:
    user_input = input("\nStudent: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    answer = chat(user_input)
    print("LLM:", answer)      
