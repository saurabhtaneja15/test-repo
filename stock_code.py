from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config.settings import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME

# Prepare the prompt template
system_prompt = (
    "You are a financial assistant. Given a company name, return ONLY its primary stock ticker symbol (e.g., 'AAPL' for Apple Inc.). "
    "If you don't know, return 'UNKNOWN'."
)
prompt_temp = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "Company name: {company_name}")
])

# Set up the model and parser
model = AzureChatOpenAI(
    model=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)
parser = StrOutputParser()

# Compose the chain
ticker_chain = prompt_temp | model | parser

def get_stock_code(company_name: str) -> str:
    """
    Use an LLM to get the stock ticker symbol for a company name.
    """
    result = ticker_chain.invoke({"company_name": company_name})
    return result.strip().upper()


# import yfinance as yf

# def get_stock_code(company_name: str) -> str:
#     """
#     Given a company name, attempt to find its stock ticker using yfinance.
#     """
#     # yfinance doesn't provide direct search by company name, so we use a workaround
#     # In production, use a more robust search or a mapping
#     search = yf.Ticker(company_name)
#     if search.info and 'symbol' in search.info:
#         return search.info['symbol']
#     # fallback: try to use the company name as ticker
#     return company_name.upper()