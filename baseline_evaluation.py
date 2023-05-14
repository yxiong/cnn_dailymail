# Databricks notebook source
# MAGIC %md ### Set up
# MAGIC
# MAGIC Install the `rouge-score` python package maintained by Google research
# MAGIC - https://pypi.org/project/rouge-score/
# MAGIC - https://github.com/google-research/google-research/tree/master/rouge

# COMMAND ----------

# MAGIC %pip install rouge-score

# COMMAND ----------

# MAGIC %md ### Baseline rouge score
# MAGIC
# MAGIC Use the first 300 characters as the summary.

# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="/dbfs/yxiong/huggingface/")

# COMMAND ----------

val_df = dataset["validation"].to_pandas()

# COMMAND ----------

val_df

# COMMAND ----------

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rougeL_scores = []

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    target = row["highlights"]
    prediction = row["article"][:300]
    score = scorer.score(target=target, prediction=prediction)
    rouge1_scores.append(score["rouge1"].fmeasure)
    rougeL_scores.append(score["rougeL"].fmeasure)

print(f'rouge 1: {np.average(rouge1_scores)}\nrouge L: {np.average(rougeL_scores)}')
