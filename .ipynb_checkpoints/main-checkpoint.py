import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from restaurant_engine_functions import (
    force_english_google_maps, calculate_sentiment, calculate_review_date,
    generate_wordcloud, plot_bigrams, extract_top_bigrams,
    plot_sentiment_over_time, plot_star_distribution, plot_aspect_sentiment
)

# Initialize FastAPI app and templates
app = FastAPI(title="Restaurant Recommender App")
templates = Jinja2Templates(directory="templates")

# Homepage: Input URL
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Submit URL and Analyze Reviews
@app.post("/submit", response_class=HTMLResponse)
async def submit_url(request: Request, url: str = Form(...)):
    try:
        # Force English URL
        url = force_english_google_maps(url)
        reviews_data = []

        # Start WebDriver (non-headless, same as notebook)
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")  # Full browser for stability
        driver = webdriver.Chrome(options=options)

        # Open Google Reviews page
        driver.get(url)

        # Accept Cookies
        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "UywwFc-LgbsSe")]'))
            )
            accept_button.click()
        except Exception:
            print("Cookies acceptance button not found or already accepted.")

        # Locate reviews container
        try:
            scrollable_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "m6QErb") and contains(@class, "DxyBCb")]'))
            )
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=500, detail=f"Failed to locate reviews container: {str(e)}")

        # Scroll to load reviews
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

        # Extract Reviews
        try:
            review_elements = driver.find_elements(By.XPATH, '//span[@class="wiI7pd"]')
            star_elements = driver.find_elements(By.XPATH, '//span[@class="kvMYJc"]')
            date_elements = driver.find_elements(By.XPATH, '//span[@class="rsqaWe"]')

            for review, star, date in zip(review_elements, star_elements, date_elements):
                reviews_data.append({
                    "review": review.text,
                    "score": int(star.get_attribute("aria-label").split()[0]),
                    "date": date.text
                })
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=500, detail=f"Failed to extract reviews: {str(e)}")

        driver.quit()

        # Convert to DataFrame
        reviews_df = pd.DataFrame(reviews_data)
        reviews_df["sentiment_score"] = reviews_df["review"].apply(calculate_sentiment)
        reviews_df["date_of_review"] = reviews_df["date"].apply(calculate_review_date)

        # Generate Visualizations
        os.makedirs("static", exist_ok=True)
        positive_reviews = reviews_df[reviews_df["sentiment_score"] > 0]["review"].tolist()
        negative_reviews = reviews_df[reviews_df["sentiment_score"] < 0]["review"].tolist()

        generate_wordcloud(positive_reviews, "Positive Word Cloud", "static/wordcloud_positive.png")
        generate_wordcloud(negative_reviews, "Negative Word Cloud", "static/wordcloud_negative.png")

        bigrams_df = extract_top_bigrams(reviews_df["review"].tolist())
        plot_bigrams(bigrams_df, "Top Bigrams", "static/bigrams.png")

        plot_sentiment_over_time(reviews_df, "static/sentiment_over_time.png")
        plot_star_distribution(reviews_df, "static/star_distribution.png")

        aspect_avg_sentiment = {"food": 0.5, "service": 0.3, "ambiance": 0.4, "price": 0.2}
        plot_aspect_sentiment(aspect_avg_sentiment, "static/aspect_sentiment.png")

        # Calculate Summary
        avg_sentiment = reviews_df["sentiment_score"].mean()
        complaint_rate = len(reviews_df[reviews_df["sentiment_score"] < -0.05]) / len(reviews_df)

        # Render Results Page
        return templates.TemplateResponse("result.html", {
            "request": request,
            "recommendation": "Recommend" if avg_sentiment > 0.1 else "Do Not Recommend",
            "avg_sentiment": avg_sentiment,
            "complaint_rate": complaint_rate
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")