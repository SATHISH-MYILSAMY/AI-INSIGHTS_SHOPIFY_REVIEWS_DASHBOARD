import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
import numpy as np
import os

st.set_page_config(page_title="Shopify Review Insights", layout="wide")

st.title("📊 Shopify Reviews Insight Dashboard")

if not os.path.exists("sample_customer_reviews_cleaned.csv"):
    st.warning("sample_customer_reviews_cleaned.csv not found. Please upload it below.")
    uploaded_file = st.file_uploader("Upload Cleaned Reviews CSV", type=["csv"])
    if uploaded_file is not None:
        with open("sample_customer_reviews_cleaned.csv", "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded and saved! Please reload the app.")
        st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv("D:\PROJECTS\AI-INSIGHTS_SHOPIFY_REVIEWS\REVIEW_ANALYTICS\Scripts\sample_customer_reviews_cleaned.csv", parse_dates=['Timestamp'])
    return df

df = load_data()

st.sidebar.header("🔍 Filters")
countries = st.sidebar.multiselect("Select Country", options=df['Shipping Country'].dropna().unique(), default=df['Shipping Country'].dropna().unique())
ratings = st.sidebar.slider("Select Rating Range", 1, 5, (1, 5))

filtered_df = df[
    (df['Shipping Country'].isin(countries)) &
    (df['Rating'] >= ratings[0]) &
    (df['Rating'] <= ratings[1]) &
    (df['Rating'].notna())
]

st.markdown(f"### Filtered {len(filtered_df)} Reviews")

if st.button("📉 Product Categories with Most 1-Star Reviews (Canada)"):
    one_star = df[(df['Rating'] == 1) & (df['Shipping Country'] == 'Canada')]
    result = one_star['Product Category'].value_counts().head(10)
    st.bar_chart(result)

if st.button("📈 Correlation: Order Value vs Rating"):
    corr_df = df[['Order Value', 'Rating']].dropna()
    X = corr_df['Order Value'].values.reshape(-1, 1)
    y = corr_df['Rating'].values
    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    st.write(f"R² Score: {score:.2f}")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Order Value', y='Rating', data=corr_df, ax=ax)
    sns.lineplot(x=corr_df['Order Value'], y=model.predict(X), color='red', ax=ax)
    st.pyplot(fig)

if st.button("💬 Top 5 Complaints and Compliments"):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    df['sentiment'] = df['Review Content'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    top_complaints = df.nsmallest(5, 'sentiment')[['Review Content', 'sentiment']]
    top_compliments = df.nlargest(5, 'sentiment')[['Review Content', 'sentiment']]

    st.subheader("Top 5 Complaints")
    for _, row in top_complaints.iterrows():
        st.error(row['Review Content'])

    st.subheader("Top 5 Compliments")
    for _, row in top_compliments.iterrows():
        st.success(row['Review Content'])

if st.button("🚚 Fulfillment Status & Negative Reviews"):
    negative_df = df[df['Rating'] <= 2]
    status_counts = negative_df['Fulfillment Status'].value_counts()
    st.bar_chart(status_counts)

st.caption("Created by Sathish Myilsamy for Shopify Review Insights")