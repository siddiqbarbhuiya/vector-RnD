from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4o-mini")

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the following query to make it more precise and suitable for semantic search, preserving the original intent. Return only the rewritten query as a string.

    Original Query: {query}
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error rewriting query: {str(e)}")
        return query

def expand_query(query: str, criteria: dict) -> str:
    prompt = f"""Expand the following query by adding relevant synonyms or related terms based on the provided criteria. Ensure the expanded query remains concise and suitable for resume search. Return only the expanded query as a string.

    Query: {query}
    Criteria: {json.dumps(criteria)}
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error expanding query: {str(e)}")
        return query