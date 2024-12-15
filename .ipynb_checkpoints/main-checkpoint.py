import os
import re
import pandas as pd
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
from restaurant_engine_functions import (
    force_english_google_maps, calculate_sentiment, calculate_review_date,
    generate_wordcloud, extract_top_bigrams, plot_star_distribution, plot_aspect_sentiment
)

# Initialize FastAPI app and templates
app = FastAPI(title="Restaurant Recommender App")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Seaborn style for better visuals
sns.set(style="whitegrid", palette="muted", font_scale=1.2)


# Function to extract restaurant name
def extract_restaurant_name(url):
    try:
        match = re.search(r'/place/(.+?)/', url)
        if match:
            return match.group(1).replace('+', ' ')
        return "Unknown Restaurant"
    except Exception as e:
        logging.error("Error extracting restaurant name: %s", e)
        return "Unknown Restaurant"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit_url(request: Request, url: str = Form(...)):
    try:
        logging.info("Received URL: %s", url)
        restaurant_name = extract_restaurant_name(url)
        url = force_english_google_maps(url)
        reviews_data = []

        # Setup Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)

        driver.get(url)

        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "UywwFc-LgbsSe")]'))
            )
            accept_button.click()
        except Exception:
            logging.warning("Cookies acceptance button not found or already accepted.")

        try:
            scrollable_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "m6QErb") and contains(@class, "DxyBCb")]'))
            )
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=500, detail=f"Failed to locate reviews container: {str(e)}")

        # Scroll reviews
        last_height = 0
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(2)
            new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
            if new_height == last_height:
                retry_count += 1
            else:
                retry_count = 0
                last_height = new_height

        # Extract reviews
        review_elements = driver.find_elements(By.XPATH, '//span[@class="wiI7pd"]')
        star_elements = driver.find_elements(By.XPATH, '//span[@class="kvMYJc"]')
        date_elements = driver.find_elements(By.XPATH, '//span[@class="rsqaWe"]')

        for review, star, date in zip(review_elements, star_elements, date_elements):
            try:
                score = int(star.get_attribute("aria-label").split()[0])
                reviews_data.append({
                    "review": review.text,
                    "score": score,
                    "date": date.text
                })
            except Exception as e:
                logging.error("Error parsing review: %s", e)

        driver.quit()

       # Process data
        reviews_df = pd.DataFrame(reviews_data)
        reviews_df["sentiment_score"] = reviews_df["review"].apply(calculate_sentiment)

        # Apply calculate_review_date and ensure datetime conversion
        reviews_df["date_of_review"] = reviews_df["date"].apply(calculate_review_date)
        reviews_df["date_of_review"] = pd.to_datetime(reviews_df["date_of_review"], errors='coerce')  # Force datetime conversion

        # Remove rows with invalid or future dates
        today = datetime.now()
        reviews_df = reviews_df.dropna(subset=["date_of_review"])  # Drop invalid dates
        reviews_df = reviews_df[reviews_df["date_of_review"] <= today]  # Remove future dates

        # Convert to Year-Month for plotting
        reviews_df["year_month"] = reviews_df["date_of_review"].dt.to_period("M").dt.to_timestamp()

        os.makedirs("static", exist_ok=True)

        # Word Clouds
        positive_reviews = reviews_df[reviews_df["sentiment_score"] > 0]["review"].tolist()
        generate_wordcloud(positive_reviews, "Positive Word Cloud", "static/wordcloud_positive.png")

        negative_reviews = reviews_df[reviews_df["sentiment_score"] < 0]["review"].tolist()
        generate_wordcloud(negative_reviews, "Negative Word Cloud", "static/wordcloud_negative.png")

        # Top Bigrams
        bigrams_df = extract_top_bigrams(reviews_df["review"].tolist())
        plt.figure(figsize=(10, 6))
        sns.barplot(y=bigrams_df['bigram'], x=bigrams_df['count'], palette="Blues_d", edgecolor="black")
        plt.title("Top Bigrams", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("static/bigrams.png")
        plt.close()

       # Sentiment Over Time
        sentiment_over_time = reviews_df.groupby("year_month")["sentiment_score"].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(sentiment_over_time.index, sentiment_over_time, marker='o', color='royalblue', linewidth=2)
        plt.title("Sentiment Over Time", fontsize=16, fontweight='bold')
        plt.xlabel("Year-Month", fontsize=12)
        plt.ylabel("Average Sentiment Score", fontsize=12)

        # Limit the x-axis to valid year_month range
        plt.xlim(sentiment_over_time.index.min(), sentiment_over_time.index.max())
        
        # Format x-axis as Year-Month
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.grid(visible=True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("static/sentiment_over_time.png")
        plt.close()

        # Star Ratings Distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x=reviews_df["score"], palette="YlOrBr", edgecolor="black")
        plt.title("Star Ratings Distribution", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("static/star_distribution.png")
        plt.close()

        # Aspect Sentiment
        aspect_avg_sentiment = {
            "food": reviews_df[reviews_df["review"].str.contains("food")]["sentiment_score"].mean(),
            "service": reviews_df[reviews_df["review"].str.contains("service")]["sentiment_score"].mean(),
            "ambiance": reviews_df[reviews_df["review"].str.contains("ambiance")]["sentiment_score"].mean(),
            "price": reviews_df[reviews_df["review"].str.contains("price")]["sentiment_score"].mean()
        }
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(aspect_avg_sentiment.keys()), y=list(aspect_avg_sentiment.values()), palette="Greens_d", edgecolor="black")
        plt.title("Aspect-Based Sentiment", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig("static/aspect_sentiment.png")
        plt.close()

        avg_sentiment = reviews_df["sentiment_score"].mean()
        complaint_rate = len(reviews_df[reviews_df["sentiment_score"] < -0.05]) / len(reviews_df)

        # Return result
        recommendation = "Recommend" if avg_sentiment > 0.5 else "Do Not Recommend"

        return templates.TemplateResponse("result.html", {
            "request": request,
            "restaurant_name": restaurant_name,
            "recommendation": recommendation,
            "avg_sentiment": round(avg_sentiment, 2),
            "complaint_rate": round(complaint_rate, 2)
        })

    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")