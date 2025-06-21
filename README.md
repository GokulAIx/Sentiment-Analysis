# 😊 Sentiment Analysis (TRAINED ON MOVIE REVIEWS)
A lightweight Streamlit app that classifies user reviews as Positive, Neutral, or Negative using a PyTorch model.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red?logo=streamlit)

![Model Accuracy](https://img.shields.io/badge/Accuracy-80%25-brightgreen)

👉 [Try it live - on GokulAIx](https://gokulaix-sentiment-analysis.streamlit.app/)

Dataset: IMDB DATASET ON Hugging Face

!pip install --force-reinstall datasets

from datasets import load_dataset

dataset = load_dataset("imdb")

## 5. How It Works / Model Overview

Model: 3-layer feedforward neural network

Input: Sentence embeddings (from MiniLM model)

Preprocessing: Uses all-MiniLM-L6-v2 from sentence-transformers

Output: Probability score between 0–1 (mapped to sentiment)

Accuracy: ~80%

## 6. App Features
   
Input a Review/Statement

Get prediction with confidence %

Color-coded output for Positive, Neutral, Negative.

## 7. How to Run Locally

Simple terminal steps:

# First, fork this repository to your own GitHub account

# Then, clone it from your own GitHub account

<pre> <code> ```bash 
  git clone https://github.com/&lt;your-username&gt;/Sentiment-Analysis.git 
``` </code> </pre>

cd Sentiment-Analysis

pip install -r requirements.txt

streamlit run app.py

## 8. Example Inputs

✅ Positive: “This movie was absolutely incredible and had me hooked till the end.”

❌ Negative: “The plot was weak and the acting felt forced.”


## 9. Project Structure

├── app.py  

├── Sentiment_Analysis.pth  

├── requirements.txt  

└── README.md  


## 10. Contact Info

- **Name:** P Gokul Sree Chandra  
- **Email:** polavarapugokul@gmail.com  
- **LinkedIn:** [Gokul Sree Chandra](https://www.linkedin.com/in/gokulsreechandra/)  
- **Portfolio:** [GokulAIx](https://soft-truffle-eada3e.netlify.app/)

## 11. License
# This project is licensed under the [MIT License](LICENSE).
