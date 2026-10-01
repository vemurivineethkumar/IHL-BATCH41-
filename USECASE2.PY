# use_case_2_rag.py

from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# LLM
# --------------------------------------------------

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

MODEL = "your-model-name"

# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="college_policies"
)


# --------------------------------------------------
# Add college policy documents
# --------------------------------------------------

def add_policy_documents():

    policies = [
        """
        ABC Institute Regulation 4.1:
        Students must maintain a minimum of 75% attendance
        in each course to be eligible for end-semester exams.
        """,

        """
        ABC Institute Regulation 4.2:
        Condonation up to 65% attendance is permitted only
        for verified medical emergencies.
        """,

        """
        ABC Institute Regulation 4.3:
        Students requesting medical condonation must submit
        Form Med-1 within 3 working days.
        """
    ]

    embeddings = embedding_model.encode(
        policies
    ).tolist()

    for i, policy in enumerate(policies):

        collection.upsert(
            ids=[f"policy_{i}"],
            documents=[policy],
            embeddings=[embeddings[i]]
        )


# --------------------------------------------------
# Retrieve relevant policy
# --------------------------------------------------

def retrieve_policy(question, k=2):

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


# --------------------------------------------------
# RAG Chatbot
# --------------------------------------------------

def rag_chat(question):

    documents = retrieve_policy(question)

    context = "\n\n".join(documents)

    prompt = f"""
You are an academic college assistant.

Answer the student's question ONLY using the
college policy provided below.

If the answer is not present in the policy,
say that you cannot verify it.

COLLEGE POLICY:
{context}

STUDENT QUESTION:
{question}

Give a clear and concise answer.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a college policy assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content, context


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    add_policy_documents()

    print("=== USE CASE 2: RAG CHATBOT ===")

    question = input(
        "Student: "
    )

    answer, retrieved = rag_chat(question)

    print("\n--- Retrieved Knowledge ---")
    print(retrieved)

    print("\n--- RAG Grounded Output ---")
    print(answer)
