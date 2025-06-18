---
title: technl
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: false
---

# Internship Selection Predictor

This is a BERT-based model that predicts whether a student will be selected for an internship based on their application details. The model takes into account various factors including:

- Technical experience
- Non-technical experience
- Career goals
- Location (with regional priority)
- Gender diversity

## How to Use

1. Enter your town/city
2. Select your gender identity
3. Describe why you want this internship
4. Provide details about your technical experience
5. Share your non-technical experience
6. Describe your career goals
7. Add any additional comments (optional)

The model will provide:
- A prediction (Selected/Not Selected)
- Confidence score
- Detailed reasoning for the decision

## Model Details

The model uses BERT (Bidirectional Encoder Representations from Transformers) for sequence classification. It has been fine-tuned on internship selection data and considers both technical qualifications and diversity factors in its decision-making process.

## Technical Stack

- PyTorch
- Transformers (Hugging Face)
- NLTK for text processing
- Gradio for the web interface 