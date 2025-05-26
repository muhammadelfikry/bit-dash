from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

url = "https://newsapi.org/v2/everything"

start_date = "2025-4-26"
end_date = "2025-5-26"

query = ["bitcoin", "ethereum", "dogecoin", "solana", "cardano", "ripple", "polkadot", "cryptocurrency"]

def fetch_news(query, start_date, end_date):
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data["articles"]
        df = pd.DataFrame(articles)
        print(f"Fetched {len(df)} articles for query '{query}' from {start_date} to {end_date}.")
        return df
    else:
        print(f"Failed to fetch news: {response.status_code} - {response.text}")


if __name__ == "__main__":
    all_news = []
    for q in query:
        news_df = fetch_news(q, start_date, end_date)
        if news_df is not None:
            all_news.append(news_df)

        else:
            print(f"No data returned for query '{q}'.")
    
    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        combined_news.to_csv("../data/raw/news_data.csv", index=False)
        print("News data saved to 'news_data.csv'.")
    
    else:
        print("No news data fetched.")