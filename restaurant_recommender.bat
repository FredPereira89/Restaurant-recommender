@echo off
cd /d "C:\Users\user\Documents\Python stuff\Restaurant-recommender"
call venv\Scripts\activate
uvicorn main:app --reload
pause