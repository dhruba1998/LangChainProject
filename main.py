from dotenv import load_dotenv
from langsmith.types.info_list_response import SDKVersions

load_dotenv()
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by an agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for a response given by an agent"""

    answer:str = Field(description="The agent's answer to query")
    sources: List[Source] = Field(default_factory=list,description="sources used to answer to the query")



# @tool
# def search_web(query: str) -> str:
#     """Search the web for information
#     Args:
#         query (str): The query to search the web for
#     Returns:
#         str: The search result
#     """
#     print(f"Searching the web for: {query}")
#     return tavily_client.search(query=query)


# tavily_client = TavilyClient() # Search using Tavily SDK

llm = ChatOpenRouter(model="google/gemma-4-26b-a4b-it")
# llm = ChatOpenRouter(model="deepseek/deepseek-v4.1-flash-1017")
# tools = [search_web]
tools = [TavilySearch(max_results=1)]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

def main():
    result = agent.invoke(
        {
            "messages":[HumanMessage(
                content="search the web to find 3 active job postings for a AI engineer on LinkedIn")]
                })
    print(result)


if __name__ == "__main__":
    main()
