import os
import argparse
import pandas as pd
from data_preprocessing import preprocess_data, clean_text
from model_training import train_model, evaluate_model, compare_models
from sentiment_analyzer import SentimentAnalyzer
from visualization import generate_all_visualizations

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
    
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'analyze', 'all'],
                        default='all', help='Mode of operation')
    
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'svm', 'naive_bayes', 'random_forest'],
                        default='logistic_regression', help='Model to use for training')
    
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    
    parser.add_argument('--input', type=str, default='data/IMDB Dataset.csv',
                        help='Path to input dataset CSV file')
    
    parser.add_argument('--analyze_file', type=str,
                        help='Path to file of reviews to analyze')
    
    parser.add_argument('--analyze_text', type=str,
                        help='Single review text to analyze')
    
    return parser.parse_args()

def load_kaggle_imdb_dataset(file_path):
    """
    Load the Kaggle IMDB dataset from CSV.
    
    Args:
        file_path: Path to the 'IMDB Dataset.csv' file
        
    Returns:
        pandas DataFrame with 'text' and 'sentiment' columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"Loading dataset from {file_path}...")
    # The Kaggle IMDB dataset has 'review' and 'sentiment' columns
    df = pd.read_csv(file_path)
    
    # Rename columns to match our expected format
    df = df.rename(columns={'review': 'text'})
    
    # Convert sentiment labels: 'positive' -> 1, 'negative' -> 0
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    print(f"Loaded {len(df)} reviews.")
    return df

def custom_preprocess(input_file):
    """Preprocess the Kaggle IMDB dataset."""
    # Load the dataset
    df = load_kaggle_imdb_dataset(input_file)
    
    # Clean the text
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Split and save
    from sklearn.model_selection import train_test_split
    
    # First split into training and temp (validation + test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Then split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create directory for processed data
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the processed datasets
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/validation.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Processed {len(df)} reviews.")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def main():
    """Main function to run the sentiment analysis pipeline."""
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n=== PREPROCESSING DATA ===")
        train_df, val_df, test_df = custom_preprocess(args.input)
    else:
        # Load preprocessed data if available
        try:
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            test_df = pd.read_csv('data/processed/test.csv')
        except FileNotFoundError:
            print("Preprocessed data not found. Run with --mode preprocess first.")
            return
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n=== TRAINING MODEL ===")
        if args.compare:
            print("Comparing multiple models...")
            results = compare_models(train_df, val_df, test_df)
            
            # Print comparison results
            print("\nModel Comparison Results:")
            for model_name, accuracies in results.items():
                print(f"{model_name.replace('_', ' ').title()}:")
                print(f"  Validation Accuracy: {accuracies['validation_accuracy']:.4f}")
                print(f"  Test Accuracy: {accuracies['test_accuracy']:.4f}")
                
        else:
            print(f"Training {args.model} model...")
            model, vectorizer, _, _ = train_model(train_df, val_df, args.model)
            test_accuracy, _ = evaluate_model(model, vectorizer, test_df)
            print(f"Test accuracy: {test_accuracy:.4f}")
    
    if args.mode == 'analyze' or args.mode == 'all' or args.analyze_file or args.analyze_text:
        print("\n=== ANALYZING SENTIMENT ===")
        
        # Check if model exists
        model_path = f'models/{args.model}_model.pkl'
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Train the model first using --mode train")
            return
            
        # Create analyzer
        analyzer = SentimentAnalyzer(model_path=model_path)
        
        # Analyze file if provided
        if args.analyze_file:
            if not os.path.exists(args.analyze_file):
                print(f"File not found: {args.analyze_file}")
            else:
                print(f"Analyzing reviews from {args.analyze_file}...")
                results_df = analyzer.analyze_batch_file(args.analyze_file)
                
                # Save results
                results_df.to_csv('results/sentiment_results.csv', index=False)
                print(f"Analysis complete. Results saved to results/sentiment_results.csv")
                
                # Generate visualizations
                generate_all_visualizations(results_df)
        
        # Analyze single text if provided
        elif args.analyze_text:
            print("Analyzing single review...")
            result = analyzer.predict_sentiment(args.analyze_text)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['score']:.4f}")
        
        # If no specific analysis task, analyze the test set
        else:
            print("Analyzing test set...")
            test_df = pd.read_csv('data/processed/test.csv')
            results_df = analyzer.analyze_reviews(test_df)
            
            # Save results
            results_df.to_csv('results/sentiment_results.csv', index=False)
            print(f"Analysis complete. Results saved to results/sentiment_results.csv")
            
            # Generate visualizations
            generate_all_visualizations(results_df)

if __name__ == "__main__":
    main()