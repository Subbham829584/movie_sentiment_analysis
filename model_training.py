import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_model(train_data, val_data, model_type='logistic_regression'):
    """
    Train a sentiment analysis model using the specified algorithm.
    """
    print(f"Training {model_type} model...")
    
    # Extract features and labels
    X_train = train_data['cleaned_text']
    y_train = train_data['sentiment']
    
    X_val = val_data['cleaned_text']
    y_val = val_data['sentiment']
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Choose the classifier based on model_type
    if model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        model = LinearSVC(random_state=42)
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Validate the model
    y_val_pred = model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Classification report:\n{report}")
    
    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open(f'models/{model_type}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, vectorizer, accuracy, report

def evaluate_model(model, vectorizer, test_data):
    """Evaluate the model on test data and create visualizations."""
    X_test = test_data['cleaned_text']
    y_test = test_data['sentiment']
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Predict
    y_test_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Classification report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # If the model has feature importances, plot them
    if hasattr(model, 'coef_'):
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients or feature importances
        if model.__class__.__name__ == 'LogisticRegression' or model.__class__.__name__ == 'LinearSVC':
            coefs = model.coef_[0]
        else:
            return accuracy, report
        
        # Create a DataFrame for the top features
        top_features = pd.DataFrame({
            'feature': feature_names,
            'importance': coefs
        })
        
        # Sort by absolute importance
        top_features['abs_importance'] = abs(top_features['importance'])
        top_features = top_features.sort_values('abs_importance', ascending=False)
        
        # Plot top positive and negative features
        plt.figure(figsize=(12, 8))
        
        # Top positive features
        pos_features = top_features[top_features['importance'] > 0].head(15)
        plt.subplot(1, 2, 1)
        sns.barplot(x='importance', y='feature', data=pos_features, palette='Blues_d')
        plt.title('Top Positive Features')
        plt.tight_layout()
        
        # Top negative features
        neg_features = top_features[top_features['importance'] < 0].head(15)
        plt.subplot(1, 2, 2)
        sns.barplot(x='importance', y='feature', data=neg_features, palette='Reds_d')
        plt.title('Top Negative Features')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('results/feature_importance.png')
        plt.close()
    
    return accuracy, report

def compare_models(train_data, val_data, test_data):
    """Train and compare multiple models."""
    models = [
        'logistic_regression',
        'svm',
        'naive_bayes',
        'random_forest'
    ]
    
    results = {}
    
    for model_type in tqdm(models, desc="Training models"):
        model, vectorizer, val_accuracy, _ = train_model(train_data, val_data, model_type)
        test_accuracy, _ = evaluate_model(model, vectorizer, test_data)
        results[model_type] = {
            'validation_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        }
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models_list = list(results.keys())
    val_accuracies = [results[m]['validation_accuracy'] for m in models_list]
    test_accuracies = [results[m]['test_accuracy'] for m in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    plt.bar(x - width/2, val_accuracies, width, label='Validation Accuracy')
    plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(x, [m.replace('_', ' ').title() for m in models_list])
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Load the preprocessed data
    train_data = pd.read_csv('data/processed/train.csv')
    val_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Train and evaluate individual model
    model, vectorizer, _, _ = train_model(train_data, val_data)
    evaluate_model(model, vectorizer, test_data)
    
    # Or compare multiple models
    # compare_models(train_data, val_data, test_data)