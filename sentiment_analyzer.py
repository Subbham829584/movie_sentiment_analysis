import os
import pickle
import pandas as pd
from data_preprocessing import clean_text

class SentimentAnalyzer:
    """Class for analyzing sentiment of movie reviews."""
    
    def __init__(self, model_path='models/logistic_regression_model.pkl', 
                 vectorizer_path='models/tfidf_vectorizer.pkl'):
        """
        Initialize the sentiment analyzer with the trained model and vectorizer.
        
        Args:
            model_path: Path to the trained model pickle file
            vectorizer_path: Path to the TF-IDF vectorizer pickle file
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text.
        
        Args:
            text: Raw text of a movie review
            
        Returns:
            dict: Contains sentiment label, score, and cleaned text
        """
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Transform to TF-IDF features
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Predict
        sentiment = self.model.predict(text_tfidf)[0]
        
        # Get probability score if available
        if hasattr(self.model, 'predict_proba'):
            score = self.model.predict_proba(text_tfidf)[0][sentiment]
        else:
            # For models like SVM that don't have predict_proba
            decision = self.model.decision_function(text_tfidf)[0]
            score = abs(decision) / 10  # Normalize
            score = min(score, 1.0)  # Cap at 1.0
        
        return {
            'sentiment': 'positive' if sentiment == 1 else 'negative',
            'score': float(score),
            'cleaned_text': cleaned_text
        }
    
    def analyze_reviews(self, reviews):
        """
        Analyze multiple reviews.
        
        Args:
            reviews: List of review texts or a pandas DataFrame with a 'text' column
            
        Returns:
            pandas DataFrame with sentiment analysis results
        """
        if isinstance(reviews, pd.DataFrame):
            if 'text' not in reviews.columns:
                raise ValueError("DataFrame must contain a 'text' column")
            texts = reviews['text'].tolist()
        elif isinstance(reviews, list):
            texts = reviews
        else:
            raise ValueError("Reviews must be a list of strings or a pandas DataFrame")
        
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            result['original_text'] = text
            results.append(result)
        
        return pd.DataFrame(results)

    def analyze_batch_file(self, file_path, text_column='text'):
        """
        Analyze reviews from a CSV or text file.
        
        Args:
            file_path: Path to the CSV or text file
            text_column: Name of the column containing review text (for CSV)
            
        Returns:
            pandas DataFrame with sentiment analysis results
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in the CSV file")
            reviews = df[text_column].tolist()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reviews = f.readlines()
        else:
            raise ValueError("File must be CSV or TXT format")
        
        return self.analyze_reviews(reviews)

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Analyze a single review
    review = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
    result = analyzer.predict_sentiment(review)
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['score']:.4f}")
    
    # Analyze multiple reviews
    reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Terrible movie. Bad acting, confusing plot, and the special effects were laughable.",
        "It was okay. Nothing special but not terrible either."
    ]
    
    results_df = analyzer.analyze_reviews(reviews)
    print("\nMultiple Reviews Analysis:")
    print(results_df[['original_text', 'sentiment', 'score']])