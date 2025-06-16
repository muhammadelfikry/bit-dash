from inference.sentiment_predictor_embedding import predict
from utils.newsapi_fetcher import fetch_news
from utils.llm_text_sumarize import summarize_descriptions
from dotenv import load_dotenv
from collections import Counter
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

load_dotenv()

st.set_page_config(page_title="Bit-Dash", layout="wide")
st.title("🧠 Bit-Dash - Crypto News Sentiment")

today = pd.to_datetime("today").normalize()

min_date = today - pd.Timedelta(days=7)

# Input dari pengguna (sidebar)
with st.sidebar:
    st.subheader("🔍 Search News")
    query = st.text_input("What trends are you looking for?", value="bitcoin")
    start_date = st.date_input("Start date", value=min_date.date())
    end_date = st.date_input("End date", value=min_date.date())

# Hanya jika ada query dan key
if query:
    NEWS_API_KEY = os.getenv("API_KEY")

    if start_date > end_date:
        st.error("❌ Start date must be before or the same as end date.")
    else:
        news = fetch_news(query, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), NEWS_API_KEY)

        if news is None or news is False or news.empty:
            st.error("❌ No news data found. It may be due to API limits or no articles found.")
            st.stop()
        
        # Ambil deskripsi valid
        desc = [d for d in news["description"].values if d is not None]
        
        if desc:
            sentiment_prediction = predict(desc)
            sentiment_counts = Counter(sentiment_prediction)

            if sentiment_counts:
                df_sentiment = pd.DataFrame({
                    'Category': list(sentiment_counts.keys()),
                    'Value': list(sentiment_counts.values())
                })

                sentiment_dominance = sentiment_counts.most_common(1)[0][0]

                # Chart section
                with st.container():
                    chart_1, chart_2 = st.columns(2)
                    
                    with chart_1:
                        st.subheader("Sentiment Distribution")
                        pie_fig = px.pie(
                            df_sentiment,
                            names="Category",
                            values="Value",
                            title=f"{query.capitalize()}: {sentiment_dominance} News Dominates",
                            color="Category",
                            color_discrete_map={
                                "Positive": "#00FF9C",
                                "Neutral": "#95a5a6",
                                "Negative": "#FC2947"
                            }
                        )
                        st.plotly_chart(pie_fig)

                    with chart_2:
                        st.subheader("News Count by Sentiment")
                        bar_fig = px.bar(
                            df_sentiment,
                            x="Category",
                            y="Value",
                            color="Category",
                            color_discrete_map={
                                "Positive": "#00FF9C",
                                "Neutral": "#95a5a6",
                                "Negative": "#FC2947"
                            },
                            title="Number of News per Sentiment",
                            labels={"Category": "", "Value": ""},
                            text_auto=True
                        )

                        bar_fig.update_layout(
                            xaxis_title=None,
                            yaxis_title=None,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False, showticklabels=False)
                        )
                        st.plotly_chart(bar_fig)
                with st.container():
                    st.subheader("📝 Summary of News Descriptions")
                    try:
                        summary_text = summarize_descriptions(query, desc)
                        st.text_area("News Summary", summary_text, height=200)
                    except Exception as e:
                        st.error(f"❌ Failed to generate summary: {e}")
            else:
                st.warning("No sentiment prediction results.")
        else:
            st.warning("No news was found or all descriptions were blank.")