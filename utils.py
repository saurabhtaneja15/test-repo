def format_output(company_name: str, stock_code: str, sentiment_results: list) -> dict:
    """
    Format the final output as a JSON-serializable dict.
    """
    return {
        "company": company_name,
        "stock_code": stock_code,
        "news_sentiments": sentiment_results
    }