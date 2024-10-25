import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, \
                        AutoModelForSeq2SeqLM, AutoConfig, pipeline
from model.prediction import MultiLabelClassifier, AspectBasedSentimentModel, predict_with_models
import torch
import os

# DATA
@st.cache_data
def get_data():
    df = pd.DataFrame()
    for i in range(0, 6):
        temp_df = pd.read_csv(f'./data/split_files/split_file_{i}.csv')
        df = pd.concat([df, temp_df])

    df['final_sentiment_prediction_label'] = df['final_sentiment_prediction'].replace({-1:'negative', 0:'neutral', 1:'positive'})
    df = df[['platform', 'level', 'sentence_idx', 'comment_timestamp', 'comment', 'comment_id', 'sentence', 'entity_category', 'final_aspect_categories', 'final_sentiment_prediction']]
    df['comment_timestamp'] = pd.to_datetime(df['comment_timestamp'], format='%Y-%m-%d')
    df['comment_timestamp_week_sun'] = df['comment_timestamp'] + pd.offsets.Week(weekday=6)

    return df

@st.cache_resource
def get_models():
    ENTITY_MODEL = 'destonedbob/nusiss-election-project-entity-model-distilbert-base-cased'
    ASPECT_MODEL_DISTIL = '/mount/src/nusiss-project-presidential-absa-streamlit-ui/model/multilabel_aspect_distil_4epochs_lr3e-5_without_test_set_split_keep_same_sent_together.pth'
    ASPECT_MODEL_SEQ2SEQ = 'destonedbob/nusiss-election-project-aspect-seq2seq-model-facebook-bart-large'
    SENTIMENT_MODEL_DISTIL = '/mount/src/nusiss-project-presidential-absa-streamlit-ui/model/sentiment_model_val_acc_6162_lr4.5e-5_wtdecay_1e-4_epochs4_256_256_256_256_smoothed_weight_warmup_and_reducelr_freeze4layers.pth'
    SENTIMENT_MODEL_SEQ2SEQ = 'destonedbob/nusiss-election-project-sentiment-seq2seq-model-facebook-bart-large'
    DISTILBERT_BASE_CASED = 'distilbert-base-cased'

    model = AutoModelForSequenceClassification.from_pretrained(ENTITY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(ENTITY_MODEL)

    model = MultiLabelClassifier(num_labels=13)
    print(ASPECT_MODEL_DISTIL)
    print(os.listdir())
    model.load_state_dict(torch.load(ASPECT_MODEL_DISTIL))
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_BASE_CASED)

    model = AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL_SEQ2SEQ)
    tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_SEQ2SEQ)

    # Sentiment Model 1
    model = torch.load(SENTIMENT_MODEL_DISTIL)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_BASE_CASED)

    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_SEQ2SEQ)
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_SEQ2SEQ)


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
            (df['comment_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['comment_timestamp'] <= pd.to_datetime(filter_end_date)) &
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



def create_aspect_column_figure(side, aspect_sentiment_count_df, aspect_sentiment_row_perc_df, entity_name):
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
                                df=aspect_sentiment_count_df, 
                                plot_type='counts',
                                plot_key='counts'+entity_name)
    
    create_aspect_level_figure(side=side,
                                df=aspect_sentiment_row_perc_df, 
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
            (df['comment_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['comment_timestamp'] <= pd.to_datetime(filter_end_date)) &
            (df['platform'].isin(filter_platform)) & 
            (df['final_aspect_categories'].isin(filter_aspect)) 
        ]

        configure_aspect_page_second_half(filtered_df)
        

    else:
        configure_aspect_page_second_half(df)



# TIME SERIES
def get_sentiment_aggregated_at_timeviz_level(df):
    def get_timeviz_level_sentiment_counts(df):
        # result = pd.DataFrame(df.groupby(['comment_timestamp', 'comment_id']).final_sentiment_prediction.mean().apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'))
        result = pd.DataFrame(df.groupby(['comment_timestamp_week_sun', 'comment_id']).final_sentiment_prediction.mean().apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'))
        result = result.reset_index().rename(columns={'final_sentiment_prediction': 'sentiment', 'final_aspect_categories': 'aspect', 'comment_timestamp_week_sun': 'comment_date', 'comment_timestamp': 'comment_date'})
        result = result.groupby(['comment_date', 'sentiment']).size().reset_index(name='count')
        result['percentage'] = result.groupby('comment_date')['count'].apply(lambda x: 100 * x / x.sum()).values
        result = result[result['comment_date'] <= pd.to_datetime(END_DATE_DEFAULT)]
        return result

    def get_timeviz_level_aspect_counts(df):
        result = pd.pivot_table(df, index='comment_timestamp_week_sun', columns='final_aspect_categories', aggfunc='count')['comment'].reset_index().fillna(0)
        result = result.rename(columns={'final_sentiment_prediction': 'sentiment', 'final_aspect_categories': 'aspect', 'comment_timestamp_week_sun': 'comment_date', 'comment_timestamp': 'comment_date'})
        result = result[result['comment_date'] <= pd.to_datetime(END_DATE_DEFAULT)]
        
        result = pd.melt(result, 
                     id_vars=['comment_date'], 
                     var_name='aspect', 
                     value_name='value')
        
        result['percentage'] = (result['value'] / result.groupby('comment_date')['value'].transform('sum')) * 100
        result = result.rename(columns={'value':'count'})
        return result
    

    trump_df = df[df.entity_category == 'trump']
    kamala_df = df[df.entity_category == 'kamala']

    trump_timeviz_sentiment_count_df = get_timeviz_level_sentiment_counts(trump_df)
    kamala_timeviz_sentiment_count_df = get_timeviz_level_sentiment_counts(kamala_df)

    trump_timeviz_aspect_count_df = get_timeviz_level_aspect_counts(trump_df)
    kamala_timeviz_aspect_count_df = get_timeviz_level_aspect_counts(kamala_df)

    return (trump_timeviz_sentiment_count_df, kamala_timeviz_sentiment_count_df, trump_timeviz_aspect_count_df, kamala_timeviz_aspect_count_df)


def create_timeviz_column_figure(side, timeviz_sentiment_count_df, timeviz_aspect_count_df,  entity_name):
    def create_timeviz_level_figure(side, df, plot_key):
        fig = px.bar(df, 
                        x='comment_date', 
                        y='percentage', 
                        color='sentiment', 
                        title="Distribution of Sentiment Over Time",
                        labels={'percentage': 'Percentage', 'count': 'Count'},
                        hover_data={'count': True}, 
                        color_discrete_map={'negative': 'red', 'neutral': 'lightgray', 'positive': 'green'}
                        )
        side.plotly_chart(fig, use_container_width=True, key=plot_key)

    def create_timeviz_level_figure2(side, df, plot_key):
        fig = px.bar(df, 
                        x='comment_date', 
                        y='percentage', 
                        color='aspect', 
                        title="Distribution of Aspect Over Time",
                        labels={'percentage': 'Percentage', 'count': 'Count'},
                        hover_data={'count': True}, 
                        color_discrete_map={'negative': 'red', 'neutral': 'lightgray', 'positive': 'green'}
                        )
        side.plotly_chart(fig, use_container_width=True, key=plot_key)

    side.markdown(f"<h3 style='text-align: center;'>{entity_name}</h3>", unsafe_allow_html=True)
    create_timeviz_level_figure(side=side,
                                df=timeviz_sentiment_count_df, 
                                plot_key='timeviz_sentiment_perc'+entity_name)
    create_timeviz_level_figure2(side=side,
                                df=timeviz_aspect_count_df, 
                                plot_key='timeviz_aspect_perc'+entity_name)

def configure_timeviz_page(df, start, end):
    def configure_timesries_page_second_half(df):
        trump_timeviz_sentiment_count_df, kamala_timeviz_sentiment_count_df, trump_timeviz_aspect_count_df, kamala_timeviz_aspect_count_df = get_sentiment_aggregated_at_timeviz_level(df)
        left, _, right = st.columns([3, 1, 3])

        create_timeviz_column_figure(left, trump_timeviz_sentiment_count_df, trump_timeviz_aspect_count_df, 'Trump')
        create_timeviz_column_figure(right, kamala_timeviz_sentiment_count_df, kamala_timeviz_aspect_count_df,  'Kamala')



    # st.markdown('<h2 style="text-align: center;">Over Time Visualization of ABSA Results</h2>', unsafe_allow_html=True)
    st.markdown('<h2>Over Time Visualization of ABSA Results</h2>', unsafe_allow_html=True)
    st.write('Data presented in the below charts are on the comment-aspect level (i.e. aggregated sentiments to unique aspects for comments)')
    st.header("Filters")
    
    

    filter_left, filter_right = st.columns(2)

    filter_platform = filter_left.multiselect("Social Media Platform", 
                                     options=PLATFORM_OPTIONS, 
                                     key="filter_category_platform_timeseries_level")
    filter_aspect = filter_right.multiselect("Aspect", 
                                    options=ASPECT_OPTIONS,
                                    key="filter_aspect_timeseries_level")
    if not filter_aspect:
        filter_aspect = ASPECT_OPTIONS
    if not filter_platform:
        filter_platform = PLATFORM_OPTIONS

    filter_start_date = filter_left.date_input("Start Date", value=start, key="filter_start_date_timeseries_level",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    filter_end_date = filter_right.date_input("End Date", value=end, key="filter_end_date_timeseries_level",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    
    if filter_left.button('Apply Filters'):

        filtered_df = df[
            (df['comment_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['comment_timestamp'] <= pd.to_datetime(filter_end_date)) &
            (df['platform'].isin(filter_platform)) & 
            (df['final_aspect_categories'].isin(filter_aspect))
        ]

        configure_timesries_page_second_half(filtered_df)
        

    else:
        configure_timesries_page_second_half(df)
        pass


# View sample:
def configure_view_sample_page(df, start, end):
    st.markdown('<h2>Sample Text with Filters</h2>', unsafe_allow_html=True)
    st.write('On this page, you may use filters to sample 1000 records. The sampling of records are random')
    st.header("Filters")
    filter_left, filter_right = st.columns(2)

    filter_aspect = filter_left.multiselect("Aspect", 
                                    options=ASPECT_OPTIONS,
                                    key="filter_aspect_samplepage")
    filter_platform = filter_right.multiselect("Social Media Platform", options=PLATFORM_OPTIONS, key="filter_category_samplepage")
    if not filter_platform:
        filter_platform = PLATFORM_OPTIONS
    if not filter_aspect:
        filter_aspect = ASPECT_OPTIONS

    filter_start_date = filter_left.date_input("Start Date", value=start, key="filter_start_date_samplepage",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    filter_end_date = filter_right.date_input("End Date", value=end, key="filter_end_date_samplepage",  min_value=START_DATE_DEFAULT, max_value=END_DATE_DEFAULT)
    
    
    if filter_left.button('Apply Filters'):

        filtered_df = df[
            (df['comment_timestamp'] >= pd.to_datetime(filter_start_date)) &
            (df['comment_timestamp'] <= pd.to_datetime(filter_end_date)) &
            (df['platform'].isin(filter_platform)) & 
            (df['final_aspect_categories'].isin(filter_aspect)) 
        ]

        sampled_df = filtered_df[['platform', 'comment_timestamp', 'sentence', 'entity_category', 'final_aspect_categories']]
        sampled_df['comment_timestamp'] = sampled_df['comment_timestamp'].dt.strftime('%Y-%m-%d')
        sampled_df['entity_category'] = sampled_df['entity_category'].str.title()
        sampled_df['final_aspect_categories'] = sampled_df['final_aspect_categories'].str.title()
        sampled_df = sampled_df.rename(columns={
            'platform':'Social Media Platform', 
            'comment_timestamp': 'Comment Date',
            'sentence': 'Sentence',
            'entity_category': 'Entity',
            'final_aspect_categories' : 'Aspect'
        })
        st.dataframe(sampled_df.sample(1000).reset_index(drop=True), use_container_width=True)
        

    else:
        sampled_df = df[['platform', 'comment_timestamp', 'sentence', 'entity_category', 'final_aspect_categories']]
        sampled_df['comment_timestamp'] = sampled_df['comment_timestamp'].dt.strftime('%Y-%m-%d')
        sampled_df['entity_category'] = sampled_df['entity_category'].str.title()
        sampled_df['final_aspect_categories'] = sampled_df['final_aspect_categories'].str.title()
        sampled_df = sampled_df.rename(columns={
            'platform':'Social Media Platform', 
            'comment_timestamp': 'Comment Date',
            'sentence': 'Sentence',
            'entity_category': 'Entity',
            'final_aspect_categories' : 'Aspect'
        })
        st.dataframe(sampled_df.sample(1000).reset_index(drop=True), use_container_width=True)


# Model Page
def configure_try_model_page():
    def get_model_result(model_type, sentence):
        if model_type == "Entire Pipeline":
            sentence_df = pd.DataFrame([sentence], columns=['sentence'])
            result = predict_with_models(sentence_df)


        if model_type in ["Entire Pipeline", "Entity Model", "Aspect Model (DistilBert)",  "Aspect Model (Seq2Seq)", 
                            "Aspect Model (Combined)", "Sentiment Model (DistilBert)", "Sentiment Model (Seq2Seq)", 
                            "Sentiment Model (Combined)"]:
            NotImplementedError('Entity Model')
        
        if model_type == "Entire Pipeline":
            return result
        
        if model_type in ["Aspect Model (DistilBert)", "Aspect Model (Combined)"]:
            NotImplementedError('Aspect Model (DistilBert)')
        
        if model_type in ["Aspect Model (Seq2Seq)", "Aspect Model (Combined)"]:
            NotImplementedError("Aspect Model (Seq2Seq)")

        if model_type == "Aspect Model (Combined)":
            NotImplementedError("Aspect Model (Combined)")
        
        if model_type in ["Aspect Model (DistilBert)",  "Aspect Model (Seq2Seq)", 
                            "Aspect Model (Combined)"]:
            return result
        
        if model_type in ["Sentiment Model (DistilBert)", "Sentiment Model (Seq2Seq)", 
                            "Sentiment Model (Combined)"]:
            NotImplementedError('Sentiment Model (DistilBert)')
        
        if model_type in ["Sentiment Model (DistilBert)", "Sentiment Model (Seq2Seq)", 
                            "Sentiment Model (Combined)"]:
            NotImplementedError("Sentiment Model (Seq2Seq)")

        if model_type == "Sentiment Model (Combined)":
            NotImplementedError("Sentiment Model (Combined)")
        
        if model_type in ["Sentiment Model (DistilBert)", "Sentiment Model (Seq2Seq)", 
                            "Sentiment Model (Combined)"]:
            return result
        # Code for Aspect model combine

    model_type = st.selectbox(
        "Choose a model to try out:", 
        ["Entire Pipeline", "Entity Model", "Aspect Model (DistilBert)",  "Aspect Model (Seq2Seq)", 
        "Aspect Model (Combined)", "Sentiment Model (DistilBert)", "Sentiment Model (Seq2Seq)", "Sentiment Model (Combined)"]
    )

    # Text input for entering a sentence
    st.session_state.input_text = ""
    st.session_state.input_text = st.text_input("Enter a sentence:")

    # Submit button
    if st.button("Run Model") and st.session_state.input_text != '':
        with st.spinner(text='Predicting...'):
            result = get_model_result(model_type, st.session_state.input_text)
        
        
        st.write('<h4>Model chosen</h4>', unsafe_allow_html=True)
        st.write(f"{model_type}")
        st.write('<h4>Input sentence:</h4>', unsafe_allow_html=True)
        st.write(f"{st.session_state.input_text}")
        st.write('<h4>Outputs</h4>', unsafe_allow_html=True)
        st.write('<h4>Entity</h4>', unsafe_allow_html=True)
        st.write(f"{'; '.join([word.title() for word in list(set(result['entity_category'].values.tolist()))])}")
        if 'final_aspect_categories' in result.columns:
            aspect_result = ''
            for idx, row in result.iterrows():
                if idx > 0:
                    aspect_result += '; '
                aspect_result += row['entity_category'] + ' - ' + row['final_aspect_categories']
            st.write('<h4>Aspect</h4>', unsafe_allow_html=True)
            st.write(f"{aspect_result.title()}")

        if 'final_sentiment_prediction' in result.columns:
            sentiment_result = ''
            for idx, row in result.iterrows():
                if idx > 0:
                    sentiment_result += '; '
                sentiment_result += row['entity_category'] + ' - ' + row['final_aspect_categories'] + ' - ' + IDX_SENTIMENT_MAP[row['final_sentiment_prediction']]
            st.write('<h4>Sentiment</h4>', unsafe_allow_html=True)
            st.write(f"{sentiment_result.title()}")
            




# COMMON UI 
def configure_sidebar(df):
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("## View a page:", ("Entity Level", "Aspect Level", "Over Time Visualization", "Try out models", "View Sample Data"))
    
    st.markdown('<h1 style="text-align: center;"> ABSA of US Elections 2024</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">This analysis is based on static data scraped from top posts between 22 Jul 24 and 15 Sep 24.<p>', unsafe_allow_html=True)

    if page == "Entity Level":
        configure_overall_entity_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
    elif page == "Aspect Level":
        configure_aspect_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
    elif page == "Over Time Visualization":
        configure_timeviz_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
    elif page == "View Sample Data":
        configure_view_sample_page(df, START_DATE_DEFAULT, END_DATE_DEFAULT)
    elif page == "Try out models":
        configure_try_model_page()

# # # Optionally, display specific aspects or sentiment scores
# aspect = st.selectbox('Select Aspect', df['final_aspect_categories'].unique())
# filtered_data = df[df['final_aspect_categories'] == aspect]
# st.write(filtered_data)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    with st.spinner(text="Downloading Models...May take up to 10 minutes"):
        get_models()
    print(f'Cuda is available: {torch.cuda.is_available()}')
    IDX_SENTIMENT_MAP = {-1: 'Negative', 0: 'Neutral', 1:'Positive'}
    ORIGINAL_DF = get_data()
    START_DATE_DEFAULT = datetime.date(2024, 7, 22)
    END_DATE_DEFAULT = datetime.date(2024, 9, 15)
    ASPECT_OPTIONS = ['campaign', 'communication', 'competence', 'controversies',
                        'ethics and integrity', 'leadership', 'personality trait', 'policies',
                        'political ideology', 'public image', 'public service record',
                        'relationships and alliances', 'voter sentiment', 'others']
    PLATFORM_OPTIONS = ['Reddit', 'Youtube']
    ENTITY_OPTIONS = ['Trump', 'Kamala']
    configure_sidebar(ORIGINAL_DF)




df = get_data()