import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function to ensure Google Maps URLs are in English
def force_english_google_maps(url):
    if "hl=" not in url:
        if "?" in url:
            return f"{url}&hl=en"
        else:
            return f"{url}?hl=en"
    return url

# Function to calculate review dates
def calculate_review_date(row_date):
    now = datetime.now()
    try:
        if "month" in row_date:
            months = int(row_date.split()[0]) if "a month ago" not in row_date else 1
            date = now - timedelta(days=30 * months)
        elif "year" in row_date:
            years = int(row_date.split()[0]) if "a year ago" not in row_date else 1
            date = now - timedelta(days=365 * years)
        elif "week" in row_date:
            weeks = int(row_date.split()[0]) if "a week ago" not in row_date else 1
            date = now - timedelta(weeks=weeks)
        elif "day" in row_date:
            days = int(row_date.split()[0]) if "a day ago" not in row_date else 1
            date = now - timedelta(days=days)
        else:
            date = now  # Default fallback

        # Ensure no future dates
        date = min(date, now)
        return date.date()  # Return only the date part
    except (ValueError, IndexError):
        return now.date()  # Return today's date as fallback

# Function for sentiment analysis using VADER
def calculate_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

# Function to extract top bigrams
def extract_top_bigrams(reviews, n=10):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    bigrams_matrix = vectorizer.fit_transform(reviews)
    bigrams = vectorizer.get_feature_names_out()
    counts = bigrams_matrix.toarray().sum(axis=0)
    bigrams_df = pd.DataFrame({'bigram': bigrams, 'count': counts})
    return bigrams_df.sort_values(by='count', ascending=False).head(n)

# Function to plot bigrams
def plot_bigrams(bigrams_df, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.barh(bigrams_df['bigram'], bigrams_df['count'], color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Bigrams', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to generate word clouds with optional colormap
def generate_wordcloud(reviews, title, save_path, colormap='viridis'):
    text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot Sentiment Over Time
def plot_sentiment_over_time(reviews_df, save_path):
    reviews_df = reviews_df.dropna(subset=["date_of_review"])
    reviews_df["date_of_review"] = pd.to_datetime(reviews_df["date_of_review"])
    sentiment_over_time = reviews_df.groupby("date_of_review")["sentiment_score"].mean()
    plt.figure(figsize=(10, 6))
    sentiment_over_time.plot(title="Sentiment Over Time", marker='o', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot Star Ratings Distribution
def plot_star_distribution(reviews_df, save_path):
    plt.figure(figsize=(10, 6))
    reviews_df["score"].value_counts().sort_index().plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Star Ratings Distribution")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot Aspect-Based Sentiment
def plot_aspect_sentiment(aspect_sentiments, save_path):
    plt.figure(figsize=(10, 6))
    aspects = list(aspect_sentiments.keys())
    scores = list(aspect_sentiments.values())
    plt.bar(aspects, scores, color="green", edgecolor="black")
    plt.title("Aspect-Based Sentiment")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()