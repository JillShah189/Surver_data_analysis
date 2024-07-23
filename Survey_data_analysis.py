import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords.zip')
        logger.info("NLTK stopwords data is already downloaded.")
    except LookupError:
        logger.info("NLTK stopwords data not found. Downloading...")
        nltk.download('stopwords')
        logger.info("NLTK stopwords data downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")

download_nltk_data()o

# Load the Excel file
file_path = 'Feedback_Resposne.xlsx'
df = pd.read_excel(file_path)

# Fill missing values
df.fillna(value='Unknown', inplace=True)

# Get the list of unique questions
questions = df['Question'].unique()

# Determine the first four and last two questions
first_four_questions = questions[:4]
last_two_questions = questions[-2:]

# Preprocessing function
lemmatizer = WordNetLemmatizer()

#nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = str(text)  # Ensure text is a string
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def classify_sentiment(text):
    text = text.lower()
    
    # Positive Responses
    if 'elaborated the subject matter' in text or 'detailed explanation' in text or 'clear explanation' in text:
        return 'Positive'
    
    # Negative Responses
    elif 'lost marks' in text or 'manual correction' in text or 'restricted to certain points' in text or 'less grades' in text or 'cannot understand handwriting' in text:
        return 'Negative'
    
    # Neutral Responses
    elif get_sentiment(text) >= 0.1:
        return 'Positive'
    elif get_sentiment(text) <= -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Sidebar for selecting the question
st.sidebar.header("Survey Question Selection")
selected_question = st.sidebar.selectbox(
    "Select a question:",
    questions,
    format_func=lambda x: x[:60]  # Display truncated question text
)

# Filter the DataFrame for the selected question
selected_df = df[df['Question'] == selected_question]

# Preprocess responses for last two questions
if selected_question in last_two_questions:
    selected_df['Processed_Responses'] = selected_df['Response'].apply(preprocess_text)
    selected_df['Sentiment_Label'] = selected_df['Response'].apply(classify_sentiment)
    selected_df['Sentiment'] = selected_df['Processed_Responses'].apply(get_sentiment)

# App title and description
st.title("Interactive Survey Data Analysis App")
st.markdown("""
This Streamlit app presents an analysis of survey responses collected from students across various subjects.
The app includes interactive visualizations such as pie charts for response distributions and detailed sentiment analysis and word clouds for selected questions.
""")

# Sidebar for choosing the type of analysis
st.sidebar.header("Analysis Options")
if selected_question in first_four_questions:
    analysis_type = 'Interactive Pie Charts'
else:
    analysis_type = st.sidebar.radio(
        "Select the analysis type:",
        ('Word Cloud', 'Sentiment Analysis')
    )

# Function to plot interactive pie charts for responses
def plot_interactive_pie_charts(df, question):
    st.subheader(f"Interactive Pie Charts for: {question}")
    
    # Prepare data for each subject
    for subject in df['Subject'].unique():
        data = df[(df['Question'] == question) & (df['Subject'] == subject)]['Response'].value_counts().reset_index()
        data.columns = ['Response', 'Count']
        
        # Create interactive pie chart using Plotly
        fig = px.pie(
            data, values='Count', names='Response',
            title=f'Responses for {subject}',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3  # Donut chart style
        )
        
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(title_x=0.5, showlegend=True)
        st.plotly_chart(fig)

# Plot interactive pie charts for the selected question
if analysis_type == 'Interactive Pie Charts' and selected_question in first_four_questions:
    plot_interactive_pie_charts(df, selected_question)

# Word Cloud and Sentiment Analysis for the last two questions
elif selected_question in last_two_questions:
    # Filter out 'Unknown' responses
    filtered_df = selected_df[selected_df['Response'] != 'Unknown']

    # Word Cloud
    if analysis_type == 'Word Cloud':
        st.subheader(f"Word Cloud for Responses to: {selected_question}")

        # Generate word cloud for each sentiment category
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            responses = filtered_df[filtered_df['Sentiment_Label'] == sentiment]['Processed_Responses']
            if not responses.empty:
                combined_text = ' '.join(responses.tolist())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
                st.write(f"**Word Cloud for {sentiment} Responses**")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                st.write(f"**Examples of {sentiment} Responses**")
                examples = filtered_df[filtered_df['Sentiment_Label'] == sentiment]['Response'].sample(n=min(5, len(responses)))
                for example in examples:
                    st.write(f"- {example}")

    # Sentiment Analysis
    elif analysis_type == 'Sentiment Analysis':
        st.subheader(f"Sentiment Analysis for: {selected_question}")

        # Display sentiment distribution
        fig = px.histogram(
            filtered_df,
            x='Sentiment',
            nbins=20,
            title=f'Sentiment Distribution for {selected_question}',
            labels={'Sentiment': 'Sentiment Polarity'},
            color_discrete_sequence=['skyblue']
        )
        st.plotly_chart(fig)
        
        st.write(f"**Average Sentiment Polarity**: {filtered_df['Sentiment'].mean():.2f}")
        st.write(f"**Number of Positive Responses**: {(filtered_df['Sentiment'] > 0).sum()}")
        st.write(f"**Number of Negative Responses**: {(filtered_df['Sentiment'] < 0).sum()}")
        st.write(f"**Number of Neutral Responses**: {(filtered_df['Sentiment'] == 0).sum()}")
