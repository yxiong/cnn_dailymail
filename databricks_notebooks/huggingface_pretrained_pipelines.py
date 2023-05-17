# Databricks notebook source
# MAGIC %md ### Set up

# COMMAND ----------

MODEL = "facebook/bart-large-cnn"
OUTPUT_CSV = "/dbfs/yxiong/cnn_dailymail/fb-bart-large-cnn.csv"

# COMMAND ----------

# MAGIC %md ### Load Dataset

# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="/dbfs/yxiong/huggingface/")
val_df = dataset["validation"].to_pandas()

# COMMAND ----------

# MAGIC %md ### Load Pipeline

# COMMAND ----------

from transformers import pipeline
summarizer = pipeline("summarization", model=MODEL, device="cuda:0")

# COMMAND ----------

# MAGIC %md ### Evaluate

# COMMAND ----------

from tqdm import tqdm

predictions = {}

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    predictions[row["id"]] = summarizer(row["article"], truncation=True)[0]["summary_text"]

# COMMAND ----------

import pandas as pd

predictions_df = pd.DataFrame([
    {"id": k, "prediction": v}
    for k,v in predictions.items()
])
predictions_df.to_csv(OUTPUT_CSV, index=False)

# COMMAND ----------

# MAGIC %md ### Evaluation with ROGUE score

# COMMAND ----------

# MAGIC %pip install rouge-score

# COMMAND ----------

import pandas as pd

pred_df = pd.read_csv(OUTPUT_CSV)
pred = {}
for _, row in pred_df.iterrows():
    pred[row["id"]] = row["prediction"]

# COMMAND ----------

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rougeL_scores = []

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    target = row["highlights"]
    prediction = pred[row["id"]]
    score = scorer.score(target=target, prediction=prediction)
    rouge1_scores.append(score["rouge1"].fmeasure)
    rougeL_scores.append(score["rougeL"].fmeasure)

print(f'rouge 1: {np.average(rouge1_scores)}\nrouge L: {np.average(rougeL_scores)}')
