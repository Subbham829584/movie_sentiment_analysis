Movie Review Sentiment Analysis
This project performs sentiment analysis on movie reviews to classify them as positive or negative. It uses machine learning techniques to analyze the text content of reviews and predict the sentiment.

Features
Data preprocessing and cleaning
Multiple machine learning models for sentiment classification:
Logistic Regression
Support Vector Machines (SVM)
Naive Bayes
Random Forest
Sentiment analysis of new reviews
Visualization of results
Command-line interface for easy use
Project Structure
movie_sentiment_analysis/
├── data/                    # For storing movie review data
│   └── processed/           # Processed datasets
├── models/                  # Saved trained models
├── results/                 # Visualization and analysis results
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── sentiment_analyzer.py
│   └── visualization.py
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
└── main.py                  # Main execution script
Setup Instructions
1. Clone the repository
bash
git clone https://github.com/yourusername/movie-sentiment-analysis.git
cd movie-sentiment-analysis
2. Set up a virtual environment (optional but recommended)
bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Download the dataset
Download the IMDB Dataset of 50K Movie Reviews from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the IMDB Dataset.csv file in the data/ directory.

Usage
Preprocessing the data
bash
python main.py --mode preprocess
Training a model
bash
# Train with default model (Logistic Regression)
python main.py --mode train

# Train with a specific model
python main.py --mode train --model svm

# Compare multiple models
python main.py --mode train --compare
Analyzing sentiment
bash
# Analyze the test set
python main.py --mode analyze

# Analyze a specific file of reviews
python main.py --analyze_file path/to/reviews.csv

# Analyze a single review
python main.py --analyze_text "This movie was fantastic! I loved the acting and the plot."
Run the complete pipeline
bash
python main.py --mode all
Models
The project includes the following machine learning models for sentiment classification:

Logistic Regression: Default model, offers a good balance of performance and speed.
Support Vector Machine (SVM): Performs well with text classification tasks.
Naive Bayes: Fast and efficient, especially for text classification.
Random Forest: Ensemble method that can capture complex patterns.
Visualizations
The project generates various visualizations to help understand the sentiment analysis results:

Sentiment distribution
Confidence score histograms
Word clouds for positive and negative reviews
Common words in positive and negative reviews
Review length distributions
All visualizations are saved in the results/ directory.

Requirements
Python 3.8+
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
tqdm
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The IMDB Dataset from Kaggle
NLTK library for natural language processing
scikit-learn for machine learning algorithms
