---
title: Amazon Review Sentiment Analyzer
emoji: 🛍️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# 🛍️ Amazon Review Sentiment Analyzer

An AI-powered web app that analyzes Amazon customer reviews and predicts their sentiment (1-5 stars) using a multilingual DistilBERT model.

## Features
- 📝 Single-review analysis with instant star prediction
- 📊 Bulk CSV analysis with dashboard
- 🎯 Performance metrics (accuracy, F1-score, confusion matrix) when ground-truth labels are provided
- 🌍 Supports English, French, German, Spanish, Italian, and Dutch

## Built With
- [Streamlit](https://streamlit.io) for the UI
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for the AI
- [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) model