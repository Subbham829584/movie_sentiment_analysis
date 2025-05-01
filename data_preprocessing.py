import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('movie_reviews')  # This is optional if you're using external data

def load_imdb_dataset():
    """
    Load the IMDB dataset from Kaggle CSV file.
    Returns pandas DataFrame with 'text' and 'sentiment' columns.
    """
    data_path = os.path.join('data', 'IMDB Dataset.csv')
    if os.path.exists(data_path):
        # Load from Kaggle IMDB dataset
        df = pd.read_csv(data_path)
        
        # Rename columns to match expected format
        df = df.rename(columns={'review': 'text'})
        
        # Convert sentiment labels: 'positive' -> 1, 'negative' -> 0
        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        
        return df
    else:
        # Alternative: Try to load from NLTK
        try:
            from nltk.corpus import movie_reviews
            
            reviews = []
            for category in ['pos', 'neg']:
                for fileid in movie_reviews.fileids(category):
                    text = ' '.join(movie_reviews.words(fileid))
                    reviews.append({
                        'text': text,
                        'sentiment': 1 if category == 'pos' else 0
                    })
            
            return pd.DataFrame(reviews)
        except:
            raise Exception("Movie review dataset not found. Please download the IMDB Dataset.csv from Kaggle and place it in the data folder.")

def clean_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def split_dataset(df, test_size=0.2, val_size=0.1):
    """Split dataset into training, validation and test sets."""
    # First split into training and temp (validation + test)
    train_df, temp_df = train_test_split(df, test_size=(test_size+val_size), random_state=42)
    
    # Then split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size+val_size), random_state=42)
    
    return train_df, val_df, test_df

def preprocess_data():
    """Main function to preprocess data."""
    # Make sure we have the necessary NLTK data
    download_nltk_data()
    
    # Load the dataset
    print("Loading dataset...")
    df = load_imdb_dataset()
    
    # Clean the text
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Split the dataset
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)
    
    # Save the processed datasets
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/validation.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Processed {len(df)} reviews.")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    preprocess_data()