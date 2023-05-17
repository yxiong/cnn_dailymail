# Databricks notebook source
# MAGIC %md ### Download the dataset

# COMMAND ----------

from datasets import load_dataset

# COMMAND ----------

dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="/dbfs/yxiong/huggingface/")

# COMMAND ----------

dataset

# COMMAND ----------

# MAGIC %md ### Display examples

# COMMAND ----------

display(dataset["train"].to_pandas())

# COMMAND ----------

# MAGIC %md ### Calculate stats

# COMMAND ----------

df = dataset["train"].to_pandas()

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_hist(s):
    average = s.mean()
    median = s.median()
    s.hist(grid=False, bins=30)
    plt.axvline(average, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {average:.2f}')
    plt.axvline(median, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.legend()

# COMMAND ----------

plot_hist(df["article"].apply(lambda x: len(x)))

# COMMAND ----------

highlight_len = df["highlights"].apply(lambda x: len(x))

# COMMAND ----------

plot_hist(highlight_len)

# COMMAND ----------

sum(highlight_len > 1000), highlight_len.max()
