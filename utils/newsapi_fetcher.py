from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
import os

def fetch_news(query, start_date, end_date, api_key):
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    url = "https://newsapi.org/v2/everything"

    response = requests.get(url, params=params)

    try:
        news_data = response.json()
        articles = news_data["articles"]
        df = pd.DataFrame(articles, columns=[
            "publishedAt", "title", 
            "description", "content"
        ])
        print(f"Fetched {len(df)} articles for query '{query}' from {start_date} to {end_date}.")

        return df
    
    except Exception as e:
        if response.status_code == 429 and "ratelimited" in response.text.lower():
            print("Rate limit exceeded")
            return False
        
        else:
            print(f"Error fetching news: error: {e}, status code: {response.status_code}, response text: {response.text}")
            return None

if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("API_KEY")

    start_date = "2025-5-11"
    end_date = "2025-6-11"

    query = ["bitcoin", "ethereum", "dogecoin", "solana", "cardano", 
             "ripple", "doge", "cryptocurrency", "memecoins",]

    is_rate_limited = False

    all_news = []
    for date in pd.date_range(start=start_date, end=end_date):
        if is_rate_limited:
            break
        
        date_str = date.strftime("%Y-%m-%d")
        for q in query:
            news_df = fetch_news(q, date_str, date_str, API_KEY)
            if news_df is False:
                is_rate_limited = True
                break

            elif news_df is not None:
                all_news.append(news_df)
            
            else:
                print(f"No data returned for query '{q}' on {date_str}.")

    if all_news:
        combined_news = pd.concat(all_news, ignore_index=True)
        combined_news.to_csv("./data/raw/news_data.csv", index=False)
        print("News data saved to 'news_data.csv'.")

    else:
        print("No news data fetched.")