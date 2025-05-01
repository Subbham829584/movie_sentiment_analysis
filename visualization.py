import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import numpy as np
from wordcloud import WordCloud
from collections import Counter

def create_result_directory():
    """Create directory for saving visualization results."""
    os.makedirs('results', exist_ok=True)
    return 'results'

def plot_sentiment_distribution(results_df):
    """
    Plot the distribution of positive and negative sentiments.
    
    Args:
        results_df: DataFrame with sentiment analysis results
    """
    plt.figure(figsize=(8, 6))
    sentiment_counts = results_df['sentiment'].value_counts()
    
    # Create bar plot
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    
    # Add counts as text on bars
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count/2, str(count), ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Save the plot
    results_dir = create_result_directory()
    plt.savefig(os.path.join(results_dir, 'sentiment_distribution.png'))
    plt.close()

def plot_confidence_histogram(results_df):
    """
    Plot histogram of confidence scores for positive and negative sentiments.
    
    Args:
        results_df: DataFrame with sentiment analysis results
    """
    plt.figure(figsize=(10, 6))
    
    # Separate positive and negative sentiments
    positive_scores = results_df[results_df['sentiment'] == 'positive']['score']
    negative_scores = results_df[results_df['sentiment'] == 'negative']['score']
    
    # Plot histograms
    if not positive_scores.empty:
        sns.histplot(positive_scores, color='green', label='Positive', alpha=0.6, bins=20)
    
    if not negative_scores.empty:
        sns.histplot(negative_scores, color='red', label='Negative', alpha=0.6, bins=20)
    
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save the plot
    results_dir = create_result_directory()
    plt.savefig(os.path.join(results_dir, 'confidence_histogram.png'))
    plt.close()

def generate_word_clouds(results_df):
    """
    Generate word clouds for positive and negative reviews.
    
    Args:
        results_df: DataFrame with sentiment analysis results
    """
    # Separate positive and negative reviews
    positive_texts = ' '.join(results_df[results_df['sentiment'] == 'positive']['cleaned_text'])
    negative_texts = ' '.join(results_df[results_df['sentiment'] == 'negative']['cleaned_text'])
    
    # Generate word clouds
    if positive_texts:
        positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_texts)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Positive Reviews')
        
        results_dir = create_result_directory()
        plt.savefig(os.path.join(results_dir, 'positive_wordcloud.png'))
        plt.close()
    
    if negative_texts:
        negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(negative_texts)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(negative_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Negative Reviews')
        
        results_dir = create_result_directory()
        plt.savefig(os.path.join(results_dir, 'negative_wordcloud.png'))
        plt.close()

def plot_common_words(results_df, n=15):
    """
    Plot most common words in positive and negative reviews.
    
    Args:
        results_df: DataFrame with sentiment analysis results
        n: Number of top words to display
    """
    # Separate positive and negative reviews
    positive_texts = ' '.join(results_df[results_df['sentiment'] == 'positive']['cleaned_text'])
    negative_texts = ' '.join(results_df[results_df['sentiment'] == 'negative']['cleaned_text'])
    
    # Tokenize and get word frequencies
    positive_words = word_tokenize(positive_texts)
    negative_words = word_tokenize(negative_texts)
    
    pos_freq = Counter(positive_words).most_common(n)
    neg_freq = Counter(negative_words).most_common(n)
    
    # Create DataFrames for plotting
    pos_df = pd.DataFrame(pos_freq, columns=['Word', 'Frequency'])
    neg_df = pd.DataFrame(neg_freq, columns=['Word', 'Frequency'])
    
    # Plot for positive words
    if not pos_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=pos_df, palette='viridis')
        plt.title(f'Top {n} Words in Positive Reviews')
        plt.tight_layout()
        
        results_dir = create_result_directory()
        plt.savefig(os.path.join(results_dir, 'positive_common_words.png'))
        plt.close()
    
    # Plot for negative words
    if not neg_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=neg_df, palette='magma')
        plt.title(f'Top {n} Words in Negative Reviews')
        plt.tight_layout()
        
        results_dir = create_result_directory()
        plt.savefig(os.path.join(results_dir, 'negative_common_words.png'))
        plt.close()

def visualize_review_lengths(results_df):
    """
    Visualize the distribution of review lengths for positive and negative reviews.
    
    Args:
        results_df: DataFrame with sentiment analysis results
    """
    # Calculate review lengths
    results_df['review_length'] = results_df['cleaned_text'].apply(len)
    
    # Create box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='sentiment', y='review_length', data=results_df)
    plt.title('Review Length Distribution by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Review Length (characters)')
    
    results_dir = create_result_directory()
    plt.savefig(os.path.join(results_dir, 'review_length_boxplot.png'))
    plt.close()
    
    # Create violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='sentiment', y='review_length', data=results_df)
    plt.title('Review Length Distribution by Sentiment (Violin Plot)')
    plt.xlabel('Sentiment')
    plt.ylabel('Review Length (characters)')
    
    plt.savefig(os.path.join(results_dir, 'review_length_violin.png'))
    plt.close()

def generate_all_visualizations(results_df):
    """
    Generate all visualizations for the sentiment analysis results.
    
    Args:
        results_df: DataFrame with sentiment analysis results
    """
    print("Generating visualizations...")
    
    try:
        # Basic sentiment distribution
        plot_sentiment_distribution(results_df)
        print("Sentiment distribution plot created.")
        
        # Confidence score histogram
        plot_confidence_histogram(results_df)
        print("Confidence histogram created.")
        
        # Word clouds
        generate_word_clouds(results_df)
        print("Word clouds created.")
        
        # Common words
        plot_common_words(results_df)
        print("Common words plots created.")
        
        # Review length analysis
        visualize_review_lengths(results_df)
        print("Review length visualizations created.")
        
        print(f"All visualizations saved to the results directory.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    # Example usage
    try:
        # Check if there are analyzed results
        results_path = 'results/sentiment_results.csv'
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            generate_all_visualizations(results_df)
        else:
            print(f"Results file not found: {results_path}")
            print("Please run the sentiment analyzer first to generate results.")
    except Exception as e:
        print(f"Error: {e}")