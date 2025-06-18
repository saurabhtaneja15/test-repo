from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json

from config.settings import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME

system_prompt = (
    "You are a financial news analyst. For the given news article about a company, "
    "extract the following as a JSON object with these fields: "
    "'company_name', 'stock_code', 'news_description', 'sentiment', 'people_names', "
    "'places_names', 'other_companies_referred', 'related_industries', "
    "'market_implications', 'confidence_score'.\n"
    "If a field is not present, use an empty string or empty list as appropriate.\n"
    "Respond ONLY with the JSON object.\n"
)

prompt_temp = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "Company name: {company_name}\nStock code: {stock_code}\nTitle: {title}\nContent: {content}")
])

model = AzureChatOpenAI(
    model=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
parser = StrOutputParser()

sentiment_chain = prompt_temp | model | parser

def analyze_sentiment(news_list: list, company_name: str, stock_code: str) -> list:
    """
    Analyze news articles using Azure OpenAI (gpt-4o-mini) and extract structured info.
    Returns a list of dicts with all required fields.
    """
    results = []
    for article in news_list:
        prompt_vars = {
            "company_name": company_name,
            "stock_code": stock_code,
            "title": article.get("title", ""),
            "content": article.get("content", "")
        }
        response = sentiment_chain.invoke(prompt_vars)
        try:
            data = json.loads(response)
        except Exception:
            # fallback: return minimal structure if parsing fails
            data = {
                "company_name": company_name,
                "stock_code": stock_code,
                "news_description": article.get("content", ""),
                "sentiment": "",
                "people_names": [],
                "places_names": [],
                "other_companies_referred": [],
                "related_industries": [],
                "market_implications": "",
                "confidence_score": ""
            }
        results.append(data)
    return results