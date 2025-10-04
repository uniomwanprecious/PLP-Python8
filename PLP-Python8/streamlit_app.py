import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Keep seaborn, it's installed and useful for colors
from collections import Counter
import re

# --- Custom Functions (Replicating Cleaning from Data_Analysis.py) ---

# We need a function to load and clean the data since Streamlit runs this code every time.
# The cleaning logic is adapted from your Data_Analysis.py
@st.cache_data # Streamlit Caching decorator for fast loading
def load_and_clean_data(file_path, nrows=50000):
    # 1. Load Data
    try:
        df = pd.read_csv(file_path, nrows=nrows, on_bad_lines='skip', encoding='utf8')
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame() # Return empty if fails

    # 2. Column Reduction
    cols_to_keep = ['title', 'abstract', 'publish_time', 'authors', 'journal', 'source_x', 'url']
    df_cleaned = df[cols_to_keep].copy()

    # 3. Handle Missing Data
    df_cleaned.loc[:, 'authors'] = df_cleaned['authors'].fillna('Unknown Author')
    df_cleaned.loc[:, 'journal'] = df_cleaned['journal'].fillna('Unknown Journal')
    df_cleaned.dropna(subset=['abstract'], inplace=True)

    # 4. Prepare Data for Analysis (Date Cleaning)
    df_cleaned['publish_time'] = pd.to_datetime(
        df_cleaned['publish_time'], 
        errors='coerce', 
        format='mixed'
    )
    df_cleaned.dropna(subset=['publish_time'], inplace=True)
    df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year.astype('int32')
    
    return df_cleaned

# --- Streamlit Application Layout ---

st.title("CORD-19 Research Paper Explorer") # Add title
st.write("Simple exploration of a sample of COVID-19 research papers, focusing on publication trends and key terms.") # Add description

# Path to your file
FILE_PATH = 'metadata.csv'

# Load the cleaned data
df_cleaned = load_and_clean_data(FILE_PATH)

if df_cleaned.empty:
    st.stop()

# --- Interactive Element (Year Filter) ---
min_year = int(df_cleaned['publication_year'].min())
max_year = int(df_cleaned['publication_year'].max())

year_range = st.slider(
    "Select Publication Year Range", # Add interactive element
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Apply the year filter to the data
df_filtered = df_cleaned[
    (df_cleaned['publication_year'] >= year_range[0]) & 
    (df_cleaned['publication_year'] <= year_range[1])
]

st.subheader(f"Analyzing {len(df_filtered)} papers from {year_range[0]} to {year_range[1]}")

# --- 1. Top 10 Publishing Journals ---
st.header("1. Top 10 Publishing Journals")
if not df_filtered.empty:
    # Calculation
    top_journals = df_filtered[df_filtered['journal'] != 'Unknown Journal']['journal'].value_counts().head(10)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_journals.index, y=top_journals.values, palette="viridis", ax=ax)
    ax.set_title(f'Top 10 Journals ({len(df_filtered)} papers)')
    ax.set_xlabel('Journal Name')
    ax.set_ylabel('Number of Publications')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig) # Display visualization in app
else:
    st.write("No data available for the selected year range.")


# --- 2. Total Publications Over Time ---
st.header("2. Total Publications Over Time")
if not df_filtered.empty:
    # Calculation
    publications_over_time = df_filtered['publication_year'].value_counts().sort_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    publications_over_time.plot(kind='line', marker='o', color='red', ax=ax)
    ax.set_title('Total Number of Publications Over Time (By Year)')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Publications')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig) # Display visualization in app


# --- 3. Top 10 Most Frequent Words ---
st.header("3. Top 10 Most Frequent Words")
if not df_filtered.empty:
    # Manual Stopwords list (to avoid NLTK network issues)
    manual_stop_words = set([
        'the', 'and', 'to', 'of', 'in', 'is', 'a', 'with', 'for', 'was', 'as', 
        'are', 'on', 'this', 'we', 'from', 'or', 'by', 'at', 'that', 'were', 
        'an', 'be', 'can', 'has', 'our', 'have', 'results', 'data', 'study', 
        'new', 'also', 'which', 'may', 'these', 'more', 'one', 'all', 'research',
        'used', 'paper', 'found', 'two', 'using', 'analysis', 'showed', 'been',
        'could', 'other', 'potential', 'time', 'infection', 'fig', 'figure', 'covid', 'sars', 'mers'
    ])
    
    def tokenize_and_clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return [word for word in text.split() if len(word) > 2]

    df_filtered['full_text'] = df_filtered['title'] + ' ' + df_filtered['abstract'].fillna('')
    all_words = []
    
    for text in df_filtered['full_text']:
        if isinstance(text, str):
            all_words.extend(tokenize_and_clean(text))

    filtered_words = [word for word in all_words if word not in manual_stop_words]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(10)

    words = [item[0] for item in top_words]
    counts = [item[1] for item in top_words]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, counts, color='darkred')
    ax.set_title('Top 10 Most Frequent Non-Stop Words in Titles and Abstracts')
    ax.set_xlabel('Word')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig) # Display visualization in app


# --- Display Sample Data ---
st.header("Sample of Cleaned Data")
st.dataframe(df_filtered.head(10)) # Show sample data