#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
import torch
import torch.nn as nn
from datasets import ClassLabel
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np
from datasets import load_dataset

logging.set_verbosity_info()


# 

# # Options

# In[ ]:
import argparse

parser = argparse.ArgumentParser(description='Select category of model')
parser.add_argument('category', type=str, help='category of columns')
args = parser.parse_args()

category_dict = {"유형": "type", "극성": "polarity", "시제": "tense", "확실성": "certainty"}
category = args.category
pretrained_model_name_or_path = "kykim/electra-kor-base"

# # Prepare Dataset

# In[ ]:

english_category = category_dict[category]   # type, polarity, tense, certainty
ds = load_dataset(
    "csv",
    data_files={"train": f"data/train_data_{english_category}.csv", \
                "test": f"data/validation_data_{english_category}.csv"}
)


# In[ ]:


names = list(set(ds["train"][category]))
num_labels = len(names)
cl = ClassLabel(num_classes=num_labels, names=names)
id2label = {k: v for k, v in enumerate(cl.names)}

ds = ds.cast_column(category, cl)


# # Model

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path,
    num_labels=num_labels,
    id2label=id2label
)


# # Preprocess

# In[ ]:


remove_columns = list(set(ds["train"].features) - {"input_ids", "token_type_ids", "attention_mask", category})
remove_columns


# In[ ]:


def tokenize_function(batch):
    tokens = tokenizer(batch["문장"], padding="max_length", truncation=True)
    return tokens

ds = ds.map(tokenize_function, batched=True, remove_columns=remove_columns)


# In[ ]:


ds = ds.with_format("torch")
ds = ds.rename_column(category, "labels")


# # Metrics

# In[ ]:


class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs

    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)

    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()


# In[ ]:


metrics = evaluate.combine([
    evaluate.load('accuracy'),
    ConfiguredMetric(evaluate.load('f1'), average='weighted'),
    ConfiguredMetric(evaluate.load('precision'), average='weighted'),
    ConfiguredMetric(evaluate.load('recall'), average='weighted'),
])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


# # Trainer

# In[ ]:


from datetime import datetime
now = datetime.now()
name = now.strftime("%Y-%m-%d_%H%M%S")
name

import wandb

wandb.init(
    name = name,
    tags = [category, pretrained_model_name_or_path],
    project = "huggingface",
)

# In[ ]:



training_args = TrainingArguments(
    output_dir=f'./results/{category}/{name}',          # output directory
    num_train_epochs=10,             # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    gradient_accumulation_steps=1,   # Number of updates steps to accumulate the gradients for
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    evaluation_strategy = "epoch",
    save_strategy= "epoch",
    learning_rate=1e-4,
    do_eval=True,
    logging_steps=50,
    fp16=True,
    run_name=name,
)


# In[ ]:

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()
