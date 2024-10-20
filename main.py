import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime



def get_data(path):
    df = pd.read_csv(path)
    df['final_sentiment_prediction_label'] = df['final_sentiment_prediction'].replace({-1:'negative', 0:'neutral', 1:'positive'})
    df = df[['platform', 'level', 'sentence_idx', 'post_timestamp', 'comment', 'comment_id', 'sentence', 'entity_category', 'final_aspect_categories', 'final_sentiment_prediction']]
    return df

def get_sentiment_aggregated_at_entity_level(df):
    def get_entity_level_sentiment_counts(df):
        result = pd.DataFrame(df.groupby('comment_id').final_sentiment_prediction.mean().apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral').value_counts())
        result = result.reset_index().rename(columns={'final_sentiment_prediction': 'sentiment'})
        total_comments = result['count'].sum()
        result['percentage'] = result['count'] / total_comments
        result['percentage'] = result['percentage'].apply(lambda x: round(x, 2))

        custom_order = ['negative', 'neutral', 'positive']
        # Convert the 'sentiment' column to a categorical type with the custom order
        result['sentiment'] = pd.Categorical(result['sentiment'], categories=custom_order, ordered=True)

        # Sort the DataFrame by 'sentiment'
        result = result.sort_values('sentiment')
        return result

    trump_df = df[df.entity_category == 'trump']
    kamala_df = df[df.entity_category == 'kamala']

    trump_sentiment_count_df = get_entity_level_sentiment_counts(trump_df)
    kamala_sentiment_count_df = get_entity_level_sentiment_counts(kamala_df)

    return (trump_sentiment_count_df, kamala_sentiment_count_df)


def get_aspects_aggregated_at_entity_level(df):
    def get_entity_level_aspect_counts(df):
        result = df[['comment_id', 'final_aspect_categories']].drop_duplicates(keep='first').final_aspect_categories.value_counts()
        result = pd.DataFrame(result.reset_index().rename(columns={'final_aspect_categories': 'aspect'}))
        others = result[result.aspect == 'others']
        result = result[result.aspect != 'others'].sort_values('count', ascending=True)
        result = pd.concat([others, result])
        total_comments = result['count'].sum()
        result['percentage'] = result['count'] / total_comments
        result['percentage'] = result['percentage'].apply(lambda x: round(x, 2))
        return result

    trump_df = df[df.entity_category == 'trump']
    kamala_df = df[df.entity_category == 'kamala']

    trump_aspect_count_df = get_entity_level_aspect_counts(trump_df)
    kamala_aspect_count_df = get_entity_level_aspect_counts(kamala_df)

    return (trump_aspect_count_df, kamala_aspect_count_df)



def create_entity_column_figure(side, entity_sentiment_df, entity_aspect_df, entity_name):
    def create_entity_level_figure(side, entity_sentiment_df, plot_key, y, x='percentage'):
        fig = px.bar(entity_sentiment_df, x=x, y=y, title=f"Distribution of {y.title()}", orientation='h')
        side.plotly_chart(fig, use_container_width=True, key=plot_key)
        if y == 'sentiment':
            side.markdown(f"Total comments: {entity_sentiment_df['count'].sum()}")

    side.markdown(f"<h3 style='text-align: center;'>{entity_name}</h3>", unsafe_allow_html=True)
    create_entity_level_figure(side, entity_sentiment_df, y='sentiment', plot_key=entity_name+'sentiment')
    create_entity_level_figure(side, entity_aspect_df, y='aspect', plot_key=entity_name+'aspect')

def configure_overall_entity_page(df):
    st.markdown('<h1 style="text-align: center;"> ABSA of US Elections 2024</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">This analysis is based on static data scraped from top posts between 18 Aug 24 and 15 Sep 25.<p>', unsafe_allow_html=True)

    both_sentiment_count_df = get_sentiment_aggregated_at_entity_level(df)
    both_aspect_count_df = get_aspects_aggregated_at_entity_level(df)

    st.markdown('<h2 style="text-align: center;">Entity Level ABSA Results</h2>', unsafe_allow_html=True)
    left, mid, right = st.columns([3, 1, 3])

    # Left
    trump_sentiment_count_df = both_sentiment_count_df[0]
    trump_aspect_count_df = both_aspect_count_df[0]
    create_entity_column_figure(left, trump_sentiment_count_df, trump_aspect_count_df, 'Trump')

    # Right
    kamala_sentiment_count_df = both_sentiment_count_df[1]
    kamala_aspect_count_df = both_aspect_count_df[1]
    create_entity_column_figure(right, kamala_sentiment_count_df, kamala_aspect_count_df, 'Kamala')


def configure_aspect_page(df):
    st.markdown('<h1 style="text-align: center;"> ABSA of US Elections 2024</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">This analysis is based on static data scraped from top posts between 18 Aug 24 and 15 Sep 25.<p>', unsafe_allow_html=True)

    st.markdown('<h2 style="text-align: center;">Aspect Level ABSA Results</h2>', unsafe_allow_html=True)


def configure_sidebar(df):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("## Select a page:", ("Entity Level", "Aspect Level"))
    if page == "Entity Level":
        configure_overall_entity_page(df)
    elif page == "Aspect Level":
        configure_aspect_page(df)
# # # Optionally, display specific aspects or sentiment scores
# aspect = st.selectbox('Select Aspect', df['final_aspect_categories'].unique())
# filtered_data = df[df['final_aspect_categories'] == aspect]
# st.write(filtered_data)


if __name__ == '__main__':
    df = get_data('./data/inference_sample_1k.csv')
    # df = get_data('./data/inference_after_sentiment_model2_filtered.csv')
    st.set_page_config(layout="wide")
    configure_sidebar(df)






    
    




df[['comment_id', 'final_aspect_categories']].drop_duplicates(keep='first').final_aspect_categories.value_counts().reset_index()