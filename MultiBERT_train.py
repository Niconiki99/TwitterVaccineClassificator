from transformers import BertConfig
from multimodal_transformers.model import BertWithTabular
from multimodal_transformers.model import TabularConfig

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re,os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm.auto import tqdm
import re, os
import torch
import torch.nn.functional as Functional
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer,EvalPrediction,AutoModel, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler, set_seed
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.data import load_data_from_folder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)


def calc_classification_metrics(p: EvalPrediction):
    predictions = p.predictions[0]
    pred_labels = np.argmax(predictions, axis=1)
    pred_scores = softmax(predictions, axis=1)[:, 1]
    labels = p.label_ids
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {
            "roc_auc": roc_auc_pred_score,
            "threshold": threshold,
            "pr_auc": pr_auc,
            "recall": recalls[ix].item(),
            "precision": precisions[ix].item(),
            "f1": fscore[ix].item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tp": tp.item(),
        }
    else:
        acc = (pred_labels == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=pred_labels,average = 'weighted')
        result = {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
            "mcc": matthews_corrcoef(labels, pred_labels),
        }

    return result



def tokenize_fn(batch):
    return tokenizer(batch['sentence'],padding='max_length', max_length=512, truncation=True)


tokenizer = AutoTokenizer.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0')


label_list=[0,1,2]
train_dataset, val_dataset, test_dataset = load_data_from_folder(
    DATA_DIR,
    text_cols=["sentence"],
    tokenizer=tokenizer,
    #numerical_cols="map",
    categorical_encode_type="None",
    numerical_transformer_method='none',
    label_col="label",
    label_list=label_list,
    sep_text_token_str=tokenizer.sep_token,
)


hf_config = AutoConfig.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0')
tabular_config = TabularConfig(
        combine_feat_method='text_only',  # change this to specify the method of combining tabular data
        cat_feat_dim=0,  # need to specify this
        numerical_feat_dim=0,  # need to specify this
        num_labels=3,   # need to specify this, assuming our task is binary classification
)
hf_config.tabular_config = tabular_config

model = AutoModelWithTabular.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0', config=hf_config)


training_args = TrainingArguments(
    use_cpu =True,
    output_dir=TRANSFORMERS_CACHE_DIR,
    logging_dir=TRANSFORMERS_CACHE_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    logging_steps=25,
    eval_steps=250,
)
set_seed(training_args.seed)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    #compute_metrics=compute_metrics,
    compute_metrics=calc_classification_metrics
    #compute_metrics=calc_metrics
)

%%time
trainer.train()

%%time
trainer.evaluate(eval_dataset=val_dataset)
