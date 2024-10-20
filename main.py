import pandas as pd
import streamlit as st

df = pd.read_csv('./data/inference_sample_1k.csv')
df['final_sentiment_prediction'] = df['final_sentiment_prediction'].replace({-1:'negative', 0:'neutral', 1:'positive'})
df = df[['platform', 'level', 'sentence_idx', 'post_timestamp', 'comment', 'comment_id', 'sentence', 'entity_category', 'final_aspect_categories', 'final_sentiment_prediction']]

st.title('Aspect-Based Sentiment Analysis Results')

# Display the DataFrame
st.dataframe(df)

# Optionally, display specific aspects or sentiment scores
aspect = st.selectbox('Select Aspect', df['aspect'].unique())
filtered_data = df[df['aspect'] == aspect]
st.write(filtered_data)