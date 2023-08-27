# -*- coding: utf-8 -*-
"""BTC_sentiment_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VlpvtOCMx3A9ergMR32qc099Csg8snCG
"""

pip install torch

pip install ipywidgets

pip install tqdm

from tqdm import tqdm

pip install transformers

pip install xformers

import pandas as pd
import numpy as np
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')

df1 = pd.read_csv('/content/drive/MyDrive/cryptoprediction/Bitcoin_news/test.csv')
df2 = pd.read_csv('/content/drive/MyDrive/cryptoprediction/Bitcoin_news/train.csv')
df3 = pd.read_csv('/content/drive/MyDrive/cryptoprediction/Bitcoin_news/validation.csv')

df = pd.concat([df1, df2, df3])
df

df = df.sort_values('datetime')
df

def extract_text_column(dataframe):
  text_column = dataframe["text"].tolist()
  text_matrix = np.array(text_column)
  return text_matrix

text_matrix = extract_text_column(df)

preds= []
for i in tqdm(text_matrix):
    preds.append(pipe(i))

preds = []

preds = []
for text in tqdm(df['text']):
    preds.append(pipe(text))

print(preds)

df_withsentiment = pd.DataFrame(preds)
df_withsentiment.columns = ['label', 'score']
df_withsentiment.to_csv('BTCnews_sentiment')