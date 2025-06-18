from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

def fetch_company_news(stock_code: str, num_articles: int = 5) -> list:
    """
    Fetch recent news articles for a given stock code using LangChain's YahooFinanceNewsTool.
    """
    tool = YahooFinanceNewsTool()
    news = tool.run(stock_code)
    # Return the top N articles
    return news[:num_articles]