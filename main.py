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
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from restaurant_engine_functions import (
    force_english_google_maps, calculate_sentiment, calculate_review_date,
    generate_wordcloud, extract_top_bigrams
)

# Initialize FastAPI app and templates
app = FastAPI(title="Restaurant Recommender App")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        reviews_df["date_of_review"] = pd.to_datetime(reviews_df["date"].apply(calculate_review_date), errors='coerce')

        # Filter invalid dates
        today = datetime.now()
        reviews_df = reviews_df.dropna(subset=["date_of_review"])
        reviews_df = reviews_df[reviews_df["date_of_review"] <= today]
        reviews_df["year_month"] = reviews_df["date_of_review"].dt.to_period("M").dt.to_timestamp()

        os.makedirs("static", exist_ok=True)

       # Word Clouds
        positive_reviews = reviews_df[reviews_df["sentiment_score"] > 0]["review"].tolist()
        generate_wordcloud(positive_reviews, "Positive Word Cloud", "static/wordcloud_positive.png")
    
        negative_reviews = reviews_df[reviews_df["sentiment_score"] < 0]["review"].tolist()
        generate_wordcloud(negative_reviews, "Negative Word Cloud", "static/wordcloud_negative.png", colormap="Reds")

        # Sentiment Over Time
        sentiment_over_time = reviews_df.groupby("year_month")["sentiment_score"].mean().reset_index()
        sentiment_over_time["year_month"] = sentiment_over_time["year_month"].dt.strftime("%Y-%m")  # Format Year-Month
        
        fig_sentiment = px.line(
            sentiment_over_time,
            x="year_month",
            y="sentiment_score",
            labels={"year_month": "Year-Month", "sentiment_score": "Average Sentiment Score"},
            title="Sentiment Over Time"
        )
        fig_sentiment.update_xaxes(type="category")  # Ensure Year-Month is treated as categories for proper ordering
        fig_sentiment.write_image("static/sentiment_over_time.png")

        # Star Ratings Distribution
        fig_star_dist = px.histogram(
            reviews_df, x="score", nbins=5,
            title="Star Ratings Distribution",
            labels={"score": "Star Rating", "count": "Count"}
        )
        fig_star_dist.write_image("static/star_distribution.png")

        # Aspect Sentiment
        aspect_avg_sentiment = {
            "food": reviews_df[reviews_df["review"].str.contains("food", case=False)]["sentiment_score"].mean(),
            "service": reviews_df[reviews_df["review"].str.contains("service", case=False)]["sentiment_score"].mean(),
            "ambiance": reviews_df[reviews_df["review"].str.contains("ambiance", case=False)]["sentiment_score"].mean(),
            "price": reviews_df[reviews_df["review"].str.contains("price", case=False)]["sentiment_score"].mean(),
        }
        fig_aspect = px.bar(
            x=list(aspect_avg_sentiment.keys()),
            y=list(aspect_avg_sentiment.values()),
            labels={"x": "Aspect", "y": "Average Sentiment"},
            title="Aspect-Based Sentiment"
        )
        fig_aspect.write_image("static/aspect_sentiment.png")

        # Top Bigrams
        bigrams_df = extract_top_bigrams(reviews_df["review"].tolist())
        fig_bigrams = px.bar(
            bigrams_df, x="count", y="bigram", orientation="h",
            title="Top Bigrams",
            labels={"count": "Count", "bigram": "Bigram"}
        )
        fig_bigrams.write_image("static/bigrams.png")

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