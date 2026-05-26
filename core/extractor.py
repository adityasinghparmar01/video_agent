# in this hame Actionable Items , decisions, questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY") , temperature = 0.3)

def build_chain(system_prompt : str):
    llm = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x:{"text": x}) | ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")]) | llm | StrOutputParser())

# agar apki traanscript bahut badi hai to apko pehle chunking karna padega uske baad har chunk ko summarize karna padega aur fir un summaries ko combine karke ek overall summary banani padegi. Is process me ham langchain ke RecursiveCharacterTextSplitter ka use karenge jo ki text ko specified chunk size me split karta hai with some overlap. Iske baad ham har chunk ki summary banayenge using mistral model, aur fir un summaries ko combine karke ek final summary banayenge.

def extract_action_items(transcript:str)-> str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)