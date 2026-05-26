import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

# chroma db apne aap automatically embeddings generate krta hai jab hum documents add krte hai, isliye hume manually embeddings generate krne ki zarurat nahi hai and do simila

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3,
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# build ragchain function se rag chain build hoti hai , pura database build hota hai aur retriever ready hota hai, ab hame ek function banana hai jo user ke question lega aur uske hisab se retriever se relevant chunks ko retrieve karega aur fir unhe prompt me dal ke llm ko answer generate karne ke liye bolega.
def build_rag_chain(transcript:str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store , k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    # full LCEL Rag Pipeline..  Build Rag chian

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return rag_chain   

# hame bas abhi pipeline banai hai , but isme hamne abhi tak koi bhi question m context pass nahi kiya hai, to iske liye hame ek function banana padega jo user ke question lega aur uske hisab se retriever se relevant chunks ko retrieve karega aur fir unhe prompt me dal ke llm ko answer generate karne ke liye bolega.

# load _ rag _ chain existing rags ko load krta hai , agar vector store already exist krta hai to usse load krke retriever ready krta hai aur fir rag chain build krta hai, agar vector store exist nahi krta to error throw krta hai ki pehle transcript se rag chain build kro.

def load_rag_chain():
    vector_store = load_vector_store()
    retriver = get_retriever()

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriver| RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer