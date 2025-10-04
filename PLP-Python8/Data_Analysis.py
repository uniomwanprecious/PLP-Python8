import pandas as pd

file_path = 'metadata.csv'
# Load only the first 50,000 rows for manageable exploration
ROWS_TO_LOAD = 50000 

print(f"Attempting to load the first {ROWS_TO_LOAD} rows of data...")

try:
    # Use the 'nrows' parameter to limit the data size
    df = pd.read_csv(file_path, nrows=ROWS_TO_LOAD)
    print("Data loading successful!")
    print("-" * 30)

    # --- Basic Exploration ---
    
    # Check the actual dimensions of the loaded sample
    print(f"DataFrame Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Display the first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display column types and non-null counts
    print("\nColumn Information:")
    # Using df.info() without verbose=False for the standard output, which is usually better
    df.info() 

except FileNotFoundError:
    print(f"Error: {file_path} not found. Please make sure you are running this from inside the 'PLP Python WK 8' folder.")
except Exception as e:
    print (f"An unexpected error occurred during loading: {e}")

    # --- Part 2: Data Cleaning and Preparation ---
# 1. Identify and Drop Columns with too many missing values or those not needed for analysis
# Columns to keep for the required analysis:
COLUMNS_TO_KEEP = [
    'title', 
    'abstract', 
    'publish_time', 
    'authors', 
    'journal', 
    'source_x', 
    'url'
]
df_cleaned = df[COLUMNS_TO_KEEP].copy() # Use .copy() to avoid SettingWithCopyWarning
print("\n--- Step 1: Column Reduction ---")
print(f"New DataFrame Shape after dropping columns: {df_cleaned.shape}")

# 2. Handle missing values in key columns
# Fill missing authors and journals to retain the paper row
df_cleaned['authors'].fillna('Unknown Author', inplace=True)
df_cleaned['journal'].fillna('Unknown Journal', inplace=True)

# Drop rows where the abstract is missing, as we need abstracts for word analysis
df_cleaned.dropna(subset=['abstract'], inplace=True)

print(f"New DataFrame Shape after dropping rows with missing abstract: {df_cleaned.shape}")

# 3. Prepare data for analysis (Date Conversion)
# Convert the 'publish_time' column to datetime format
df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')

# Drop any rows where the publish_time conversion failed (will become NaT)
df_cleaned.dropna(subset=['publish_time'], inplace=True)

# Extract the year from the publication date
df_cleaned['publication_year'] = df_cleaned['publish_time'].dt.year

print(f"Final DataFrame Shape after date cleaning: {df_cleaned.shape}")
print("-" * 30)

# Final check of the cleaned data
print("Cleaned Data Info:")
df_cleaned.info()

import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Part 3: Data Analysis and Visualization ---")
print("--- Task 1: Top 10 Publishing Journals ---")

# Calculate the frequency of journals, excluding our 'Unknown Journal' filler
top_journals = df_cleaned[df_cleaned['journal'] != 'Unknown Journal']['journal'].value_counts().head(10)

# Prepare for plotting
plt.figure(figsize=(12, 6))
# Use a color palette for better visual distinction
sns.barplot(x=top_journals.index, y=top_journals.values, palette="viridis") 

plt.title('Top 10 Most Frequent Publishing Journals (from 34,332 Papers)')
plt.xlabel('Journal Name')
plt.ylabel('Number of Publications')
plt.xticks(rotation=45, ha='right') # Rotate names for readability
plt.tight_layout() # Adjust layout to prevent cutting off labels
plt.show()

print("Visualization 1: Top 10 Journals completed.")

print("\n--- Task 2: Total Publications Over Time ---")

# Group publications by the extracted year
publications_over_time = df_cleaned['publication_year'].value_counts().sort_index()

# Prepare for plotting
plt.figure(figsize=(12, 6))
# Use a line plot to show trend over time
publications_over_time.plot(kind='line', marker='o', color='red') 

plt.title('Total Number of Publications Over Time (By Year)')
plt.xlabel('Publication Year')
plt.ylabel('Number of Publications')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Visualization 2: Publications Over Time completed.")

from collections import Counter
import re
import matplotlib.pyplot as plt # Keep this for the final plot

print("\n--- Task 3: Top 10 Most Frequent Words in Abstracts/Titles (Modified) ---")

# --- Manual Stopwords to bypass NLTK download error ---
# This list covers the most common English words, ensuring the analysis can complete.
manual_stop_words = set([
    'the', 'and', 'to', 'of', 'in', 'is', 'a', 'with', 'for', 'was', 'as', 
    'are', 'on', 'this', 'we', 'from', 'or', 'by', 'at', 'that', 'were', 
    'an', 'be', 'can', 'has', 'our', 'have', 'results', 'data', 'study', 
    'new', 'also', 'which', 'may', 'these', 'more', 'one', 'all', 'research',
    'used', 'paper', 'found', 'two', 'using', 'analysis', 'showed', 'been',
    'could', 'other', 'potential', 'time', 'infection', 'fig', 'figure' 
])

# 1. Combine title and abstract into one large text column
df_cleaned['full_text'] = df_cleaned['title'] + ' ' + df_cleaned['abstract']

# 2. Function to clean and tokenize text
def tokenize_and_clean(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove short words (less than 3 characters)
    words = [word for word in text.split() if len(word) > 2]
    return words

# Apply the cleaning function and flatten the list of words
all_words = []
for text in df_cleaned['full_text']:
    # Adding a simple check to skip NaN/non-string values
    if isinstance(text, str):
        all_words.extend(tokenize_and_clean(text))

# 3. Filter out stop words
filtered_words = [word for word in all_words if word not in manual_stop_words]

# 4. Count the most common words
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(10)

# Separate words and counts for plotting
words = [item[0] for item in top_words]
counts = [item[1] for item in top_words]

# 5. Visualize the Top 10 Words (Bar Chart)
plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='darkred') 

plt.title('Top 10 Most Frequent Non-Stop Words in Titles and Abstracts')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Visualization 3: Word Frequency completed. Assignment analysis complete.")