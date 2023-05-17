# Databricks notebook source
# MAGIC %md ### Set up
# MAGIC
# MAGIC Install the `rouge-score` python package maintained by Google research
# MAGIC - https://pypi.org/project/rouge-score/
# MAGIC - https://github.com/google-research/google-research/tree/master/rouge

# COMMAND ----------

# MAGIC %pip install rouge-score

# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="/dbfs/yxiong/huggingface/")

# COMMAND ----------

val_df = dataset["validation"].to_pandas()

# COMMAND ----------

from tqdm import tqdm
import utils

# COMMAND ----------

# MAGIC %md ### Compute baseline results

# COMMAND ----------

# MAGIC %md #### First 300 characters

# COMMAND ----------

predictions = {}

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    predictions[row["id"]] = row["article"][:300]

utils.write_predictions_to_csv(predictions, "/dbfs/yxiong/cnn_dailymail/first-300-chars.csv")

# COMMAND ----------

# MAGIC %md #### Entire article

# COMMAND ----------

predictions = {}

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    predictions[row["id"]] = row["article"]

# COMMAND ----------

# MAGIC %md ### Compute Rogue Score

# COMMAND ----------

import numpy as np
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0]):
    target = row["highlights"]
    prediction = predictions[row["id"]]
    score = scorer.score(target=target, prediction=prediction)
    rouge1_scores.append(score["rouge1"].fmeasure)
    rouge2_scores.append(score["rouge2"].fmeasure)
    rougeL_scores.append(score["rougeL"].fmeasure)

print(f'''
rouge 1: {np.average(rouge1_scores)}
rouge 2: {np.average(rouge2_scores)}
rouge L: {np.average(rougeL_scores)}''')
