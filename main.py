import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from pipeline.stock_code import get_stock_code
from pipeline.news_fetcher import fetch_company_news

# Azure OpenAI config
from config.settings import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME

print("Azure OpenAI Endpoint:", AZURE_OPENAI_ENDPOINT)
print("Azure OpenAI API Key:", AZURE_OPENAI_API_KEY)


# 1. Stock code extraction step (function)
def stock_code_step(inputs):
    company_name = inputs["company_name"]
    stock_code = get_stock_code(company_name)
    return {"company_name": company_name, "stock_code": stock_code}

# 2. News fetch step (function)
def news_fetch_step(inputs):
    stock_code = inputs["stock_code"]
    news = fetch_company_news(stock_code)
    return {"company_name": inputs["company_name"], "stock_code": stock_code, "news": news}

# 3. Sentiment analysis chain (LangChain style)
system_prompt = (
    "Analyze the sentiment (positive, negative, neutral) of the following news article about a company. "
    "Return only the sentiment as a single word."
)
prompt_temp = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "Title: {title}\nContent: {content}")
])
model = AzureChatOpenAI(
    model='myllm',
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
parser = StrOutputParser()

sentiment_chain = prompt_temp | model | parser

# 4. Sentiment step (function)
def sentiment_step(inputs):
    news = inputs["news"]
    sentiments = []
    for article in news:
        sentiment = sentiment_chain.invoke({
            "title": article["title"],
            "content": article["content"]
        }).strip().lower()
        sentiments.append({
            "title": article["title"],
            "content": article["content"],
            "sentiment": sentiment
        })
    return {
        "company_name": inputs["company_name"],
        "stock_code": inputs["stock_code"],
        "news_sentiments": sentiments
    }

# 5. Final formatting step (function)
def format_output_step(inputs):
    return {
        "company": inputs["company_name"],
        "stock_code": inputs["stock_code"],
        "news_sentiments": inputs["news_sentiments"]
    }

# Compose the chain using the | operator
chain = (
    RunnableLambda(stock_code_step)
    | RunnableLambda(news_fetch_step)
    | RunnableLambda(sentiment_step)
    | RunnableLambda(format_output_step)
)

def main(company_name: str):
    result = chain.invoke({"company_name": company_name})
    print(result)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <company_name>")
    else:
        main(sys.argv[1])