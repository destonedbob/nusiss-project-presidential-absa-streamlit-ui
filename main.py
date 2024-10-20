import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime



# DATA

@st.cache_data
def get_data():
    df = pd.DataFrame()
    for i in range(0, 6):
        temp_df = pd.read_csv(f'./data/split_files/split_file_{i}.csv')
        df = pd.concat([df, temp_df])

    df['final_sentiment_prediction_label'] = df['final_sentiment_prediction'].replace({-1:'negative', 0:'neutral', 1:'positive'})
    df = df[['platform', 'level', 'sentence_idx', 'post_timestamp', 'comment', 'comment_id', 'sentence', 'entity_category', 'final_aspect_categories', 'final_sentiment_prediction']]
    df['post_timestamp'] = pd.to_datetime(df['post_timestamp'], format='%Y-%m-%d')
    return df

# ENTITY LEVEL

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
            side.markdown(f"Total comments: {format(entity_sentiment_df['count'].sum(), ',')}")

    side.markdown(f"<h3 style='text-align: center;'>{entity_name}</h3>", unsafe_allow_html=True)
    create_entity_level_figure(side, entity_sentiment_df, y='sentiment', plot_key=entity_name+'sentiment')
    create_entity_level_figure(side, entity_aspect_df, y='aspect', plot_key=entity_name+'aspect')


def configure_overall_entity_page(df, start, end):
    def configure_entity_page_second_half(df):
        both_sentiment_count_df = get_sentiment_aggregated_at_entity_level(df)
        both_aspect_count_df = get_aspects_aggregated_at_entity_level(df)

        left, mid, right = st.columns([3, 1, 3])

        # Left
        trump_sentiment_count_df = both_sentiment_count_df[0]
        trump_aspect_count_df = both_aspect_count_df[0]
        create_entity_column_figure(left, trump_sentiment_count_df, trump_aspect_count_df, 'Trump')

        # Right
        kamala_sentiment_count_df = both_sentiment_count_df[1]
        kamala_aspect_count_df = both_aspect_count_df[1]
        create_entity_column_figure(right, kamala_sentiment_count_df, kamala_aspect_count_df, 'Kamala')

        
    st.markdown('<h2 style="text-align: center;">Entity Level ABSA Results</h2>', unsafe_allow_html=True)
    st.header("Filters")
    filter_platform = st.multiselect("Social Media Platform", options=['Reddit', 'Youtube'], key="filter_platform_entity")
    filter_left, filter_right = st.columns(2)
    filter_start_date = filter_left.date_input("Start Date", value=start, key="filter_start_date_entity",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    filter_end_date = filter_right.date_input("End Date", value=end, key="filter_end_date_entity",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    if not filter_platform:
        filter_platform = ['Reddit', 'Youtube']

    if filter_left.button('Apply Filters'):
        filtered_df = df[
            (df['post_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['post_timestamp'] <= pd.to_datetime(filter_end_date)) &
            (df['platform'].isin(filter_platform))
        ]
        configure_entity_page_second_half(filtered_df)

    else:
        configure_entity_page_second_half(df)



# ASPECT LEVEL

def get_sentiment_aggregated_at_aspect_level(df):
    def get_aspect_level_sentiment_counts(df):

        result = pd.DataFrame(df.groupby(['final_aspect_categories', 'comment_id']).final_sentiment_prediction.mean().apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'))
        result = result.reset_index().rename(columns={'final_sentiment_prediction': 'sentiment', 'final_aspect_categories': 'aspect'})
        result = pd.pivot_table(result, index='aspect', columns='sentiment', aggfunc='count')['comment_id'].reset_index()
        
        # Sort 
        result['total'] = result[['negative', 'neutral', 'positive']].sum(axis=1)
        result = result.sort_values(by='total', ascending=True)

        result_perc_row_wise = result.copy()
        result_perc_row_wise[['negative', 'neutral', 'positive']] =  result_perc_row_wise[['negative', 'neutral', 'positive']].div(result[['negative', 'neutral', 'positive']].sum(axis=1), axis=0) * 100
        for col in ['negative', 'neutral', 'positive']:
            result_perc_row_wise[col] = result_perc_row_wise[col].apply(lambda x: round(x,2))
        
        result_perc_row_wise = result_perc_row_wise.melt(id_vars=['aspect'], value_vars=['negative', 'neutral', 'positive'], 
                    var_name='sentiment', value_name='percentage')
        result = result.melt(id_vars=['aspect'], value_vars=['negative', 'neutral', 'positive'], 
                    var_name='sentiment', value_name='count')

        return (result, result_perc_row_wise)


    trump_df = df[df.entity_category == 'trump']
    kamala_df = df[df.entity_category == 'kamala']

    trump_result = get_aspect_level_sentiment_counts(trump_df)
    trump_aspect_sentiment_count_df = trump_result[0]
    trump_aspect_sentiment_row_perc_df = trump_result[1]

    kamala_result = get_aspect_level_sentiment_counts(kamala_df)
    kamala_aspect_sentiment_count_df = kamala_result[0]
    kamala_aspect_sentiment_row_perc_df = kamala_result[1]

    return (trump_aspect_sentiment_count_df, trump_aspect_sentiment_row_perc_df, 
             kamala_aspect_sentiment_count_df, kamala_aspect_sentiment_row_perc_df)



def create_aspect_column_figure(side, trump_aspect_sentiment_count_df, trump_aspect_sentiment_row_perc_df, entity_name):
    def create_aspect_level_figure(side, df, plot_type, plot_key, y='aspect', color='sentiment'):
        if plot_type == 'counts':
            word = 'Volume'
            x = 'count'
        elif plot_type == 'percentage':
            word = 'Proportion'
            x = 'percentage'
        fig = px.bar(df, x=x, y=y, 
                     title=f"{word} of Aspect-Comment by Sentiment", 
                     orientation='h', color=color,
                     color_discrete_map={'negative': 'red', 'neutral': 'lightgray', 'positive': 'green'}
                     )
        side.plotly_chart(fig, use_container_width=True, key=plot_key)

    side.markdown(f"<h3 style='text-align: center;'>{entity_name}</h3>", unsafe_allow_html=True)
    create_aspect_level_figure(side=side,
                                df=trump_aspect_sentiment_count_df, 
                                plot_type='counts',
                                plot_key='counts'+entity_name)
    
    create_aspect_level_figure(side=side,
                                df=trump_aspect_sentiment_row_perc_df, 
                                plot_type='percentage',
                                plot_key='percentage'+entity_name)



def configure_aspect_page(df, start, end):
    def configure_aspect_page_second_half(df):
        trump_aspect_sentiment_count_df, trump_aspect_sentiment_row_perc_df, kamala_aspect_sentiment_count_df, kamala_aspect_sentiment_row_perc_df = get_sentiment_aggregated_at_aspect_level(df)
        left, mid, right = st.columns([3, 1, 3])
        create_aspect_column_figure(left, trump_aspect_sentiment_count_df, trump_aspect_sentiment_row_perc_df, 'Trump')
        create_aspect_column_figure(right, kamala_aspect_sentiment_count_df, kamala_aspect_sentiment_row_perc_df, 'Kamala')

    st.markdown('<h2 style="text-align: center;">Aspect Level ABSA Results</h2>', unsafe_allow_html=True)
    st.header("Filters")
    filter_aspect = st.multiselect("Aspect", 
                                    options=ASPECT_OPTIONS,
                                    key="filter_aspect_aspect_level")
    filter_platform = st.multiselect("Social Media Platform", options=PLATFORM_OPTIONS, key="filter_category_platform_level")
    if not filter_platform:
        filter_platform = PLATFORM_OPTIONS
    if not filter_aspect:
        filter_aspect = ASPECT_OPTIONS
    filter_left, filter_right = st.columns(2)
    filter_start_date = filter_left.date_input("Start Date", value=start, key="filter_start_date_aspect_level",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    filter_end_date = filter_right.date_input("End Date", value=end, key="filter_end_date_aspect_level",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    
    
    if filter_left.button('Apply Filters'):

        filtered_df = df[
            (df['post_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['post_timestamp'] <= pd.to_datetime(filter_end_date)) &
            (df['platform'].isin(filter_platform)) & 
            (df['final_aspect_categories'].isin(filter_aspect)) 
        ]

        configure_aspect_page_second_half(filtered_df)
        

    else:
        configure_aspect_page_second_half(df)



# COMMON UI 
def configure_sidebar(df):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("## Select a page:", ("Entity Level", "Aspect Level"))
    
    st.markdown('<h1 style="text-align: center;"> ABSA of US Elections 2024</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">This analysis is based on static data scraped from top posts between 18 Aug 24 and 15 Sep 25.<p>', unsafe_allow_html=True)

    if page == "Entity Level":
        configure_overall_entity_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
    elif page == "Aspect Level":
        configure_aspect_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
# # # Optionally, display specific aspects or sentiment scores
# aspect = st.selectbox('Select Aspect', df['final_aspect_categories'].unique())
# filtered_data = df[df['final_aspect_categories'] == aspect]
# st.write(filtered_data)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    ORIGINAL_DF = get_data()
    START_DATE_DEFAULT = datetime.date(2024, 8, 18)
    END_DATE_DEFAULT = datetime.date(2024, 9, 15)
    ASPECT_OPTIONS = ['campaign', 'communication', 'competence', 'controversies',
                        'ethics and integrity', 'leadership', 'personality trait', 'policies',
                        'political ideology', 'public image', 'public service record',
                        'relationships and alliances', 'voter sentiment', 'others']
    PLATFORM_OPTIONS = ['Reddit', 'Youtube']
    configure_sidebar(ORIGINAL_DF)




