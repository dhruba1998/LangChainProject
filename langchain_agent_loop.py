import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage, ToolMessage
from langsmith import traceable

MAX_ITERATION=10
MODEL="qwen3:1.7b"


# ----------------tools------------------

@tool
def get_product_price(product:str) -> float:
    """Get the price of the given product from the product catelog"""
    print(f"Fetching price of {product.lower()}")
    product_catelog = {"laptop":144356.34,"headphone":244.56,"keyboard":4235.56}
    product_price = product_catelog.get(product.lower(),0)
    print(f"Price of {product} is {product_price}")
    return product_price

@tool
def apply_discount_to_product(price:float,discount_tier:str) -> float:
    """Apply the discount on the product and return the final price of the product.
    Available Tiers: Glod, Silver, Bronze
    """
    print(f"Applying {discount_tier.lower()} tier discount on price {price}")
    discount_rate_per_tier = {"gold":20,"silver":15,"bronze":5}
    discount_rate = discount_rate_per_tier.get(discount_tier.lower(),0)
    final_price = round(price*(1 - discount_rate/100),2)
    print(f"Final price {final_price}")
    return final_price

# ----------------agent------------------

@traceable(name="LangChain Agent Loop")
def agent(query:str) -> str:
    tools = [get_product_price,apply_discount_to_product]
    tools_dict = {tool.name:tool for tool in tools}
    llm = init_chat_model(model=f"ollama:{MODEL}")
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=""" 
            You are a helpful shopping assistant.
            Your task is to return the final price(after applying discount if available) of a product asked by user.
            You have the following tools to use:
            get_product_price, apply_discount_to_product
            Follow the below guide step by step to fetch and apply discount on the product
            Guide:
            1. First fetch the price of the product using get_product_price tool.
            2. Apply discount using apply_discount_to_product as per the mentioned discount tier.
            
            STRICTLY FOLLOW THE BELOW RULES:
            1. Get the product price using get_product_price tool, don't assume product price if it's not available.
            If the product is not available simply inform that the specified product is not available
            2. Always pass product and discount-tier arguments in lower case. It's true for any tool passing string argument.
            3. Before calling apply_discount_to_product always call get_product_price to get the actual price.
            4. Always apply the discount(if it's a valid tier) using apply_discount_to_product tool.
            """
        ),
        HumanMessage(content=query)
    ]
    for iteration in range(1,MAX_ITERATION+1):
        print(f"Iteration: ${iteration}")
        ai_message = llm_with_tools.invoke(messages)
        # print(ai_message)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"No more further tool call required")
            return ai_message.content

        tool_to_be_called = tool_calls[0] #Taking only the 1st tool_call, in case there are multiple tool_calls present
        tool_name = tool_to_be_called.get("name")
        tool_id = tool_to_be_called.get("id")
        tool_args = tool_to_be_called.get("args",{})
        tool = tools_dict.get(tool_name)

        if tool is None:
            raise ValueError(f"{tool_name} is not a valid tool")

        observation = tool.invoke(tool_args)
        tool_message = ToolMessage(content=str(observation),tool_call_id=tool_id)
        messages.append(ai_message)
        messages.append(tool_message)

    print(f"Max iteration reached")
    return None


if __name__ == "__main__":
    response = agent("What is the price of Laptop after applying Gold discount")
    print(response)


# [{'name': 'get_product_price', 'args': {...}, 'id': 'fcd1fd1c-d5fd-4d23-a48e-08f7b73af7c1', 'type': 'tool_call'}]

# Few observations:
# 1. LLM doesn't always follow system message. Sometimes it follow partial instruction.
# 2. In case of method arguments, case-sensitivity brings overhead. 
#    For deterministic result, always need to change the case to either lower or upper.
# 3. Following instructions depends on LLM capability, small model tends to bypass some instructions and on the other hand larger model tends follow each instructions properly but comes with more token cost.  
# 4. If you use ReAct prompt then you need to parse(Regex matching) raw text message coming from LLM to get the tool name(method name), arguments and parse(Type casting like str-> float,int) the argument type as well. 
# 5. Using ReAct prompt brings lot of edge case handling. If anyone of the edge cases is not handled then it'll break all regex matching and output will not be as expected.

