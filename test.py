import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
def load_data():
    data = pd.read_csv("Women's Clothing E-Commerce dataset.csv")
    return data

# Function for text preprocessing
def preprocess_text(text):
    if pd.isnull(text):  # Check if text is NaN
        return ""  # Return empty string for NaN values
    else:
        # Tokenization
        tokens = word_tokenize(text)
        
        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        # Join the tokens back into a single string
        processed_text = ' '.join(tokens)
        return processed_text

# Main function to run Streamlit app
def main():
    st.title("Text Preprocessing")

    # Load data
    data = load_data()

    # Apply text preprocessing
    data['Processed Text'] = data['Review Text'].apply(preprocess_text)

    # Display preprocessed text for a few reviews
    st.header("Preprocessed Text for a Few Reviews")
    sample_reviews = data['Processed Text'].sample(5)  # Getting a random sample of 5 preprocessed reviews
    for i, review in enumerate(sample_reviews):
        st.subheader(f"Review {i+1}")
        st.write(review)

    # Plot word cloud of the most common words
    st.header("Word Cloud of Most Common Words")
    wordcloud = WordCloud(width=800, height=400, background_color ='white', stopwords=None, max_words=50).generate(' '.join(data['Processed Text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
