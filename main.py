
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter

load_dotenv()

def generate_summary():

    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and past public official known for his leadership of Tesla and SpaceX. 
    Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$828 billion.
    He is the founder, CEO, and Chief Engineer of SpaceX; early investor, CEO, and Product Architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI.
    He is also the founder of X, Inc. (formerly Twitter, Inc.), which he purchased in 2022.
    Musk is a strong advocate for clean energy and sustainable transportation.
    He is also a strong advocate for human rights and civil liberties.
    He is also a strong advocate for freedom of speech.
    He is also a strong advocate for freedom of the press.
    He is also a strong advocate for freedom of religion.
    """

    system_prompt_template = PromptTemplate(
        template="""
        You are a helpful assistant that generates a summary of the following information.
        Given the {information} about a person I want you to create:
        1.A short summary
        2.Two interesting facts about them.
        """,
        input_variables=["information"]
    )

    llm = ChatOpenRouter(
        model="google/gemma-4-26b-a4b-it",
        api_key=os.getenv("OPENROUTER_API_KEY")
        )

    chain = system_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == '__main__':
    generate_summary()

