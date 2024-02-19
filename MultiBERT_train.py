""" 
The provided script is a combination of functions and main training logic for a multimodal classification model. The script is designed to perform tasks such as reading and preprocessing datasets, defining model configurations, and training a transformer model on multimodal data (text and tabular).

Libraries Used:
numpy: Fundamental package for scientific computing with Python.
scipy.special: Additional functions for mathematical operations (used for softmax).
sklearn.metrics: Metrics for evaluating classification models (e.g., ROC AUC, precision-recall curve, confusion matrix).
re, os: Regular expressions and operating system interfaces.
pandas: Data manipulation and analysis library.
tqdm.auto: A fast, extensible progress bar for loops and CLI.
torch: PyTorch library for deep learning.
transformers: Hugging Face's Transformers library for natural language processing (NLP) tasks.
multimodal_transformers: Custom module for multimodal transformers.
datasets: Hugging Face's Datasets library for easy access to datasets."""
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
import re,os
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as Functional
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer,EvalPrediction,AutoModel,DefaultDataCollator, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler, set_seed
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.data import load_data_from_folder, load_data
from sklearn.metrics import  accuracy_score, precision_score, recall_score, confusion_matrix
from configuration_params import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR




def calc_classification_metrics(p: EvalPrediction)-> dict:
    """
    Calculate various classification metrics based on the evaluation predictions.

    Parameters:
    - p (EvalPrediction): An instance of EvalPrediction containing prediction and label information.

    Returns:
    - result (dict): A dictionary containing calculated classification metrics.

    Classification Metrics:
    1. For Binary Classification:
        - roc_auc: Receiver Operating Characteristic Area Under the Curve (ROC AUC) score.
        - threshold: Threshold value that maximizes the F1 score in precision-recall curve.
        - pr_auc: Area Under the Precision-Recall Curve (PR AUC) score.
        - recall: Recall at the selected threshold.
        - precision: Precision at the selected threshold.
        - f1: F1 score at the selected threshold.
        - tn: True Negative count.
        - fp: False Positive count.
        - fn: False Negative count.
        - tp: True Positive count.

    2. For Multi-class Classification:
        - acc: Accuracy.
        - f1: Weighted F1 score.
        - acc_and_f1: Average of Accuracy and Weighted F1 score.
        - mcc: Matthews Correlation Coefficient.

    """
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
   
def read_and_convert_df(
    path: str,
    names: list[str],
    dtype: dict,
    text_cols: list[str],
    tokenizer: AutoTokenizer,
    categorical_cols: list[str],
    numerical_cols: list[str],
    categorical_encode_type: str,
    label_col: str,
    label_list: list[int]
) -> torch.utils.data.dataset:
    """
    Reads a CSV file into a DataFrame, preprocesses it, and converts it into a dataset suitable for training.

    Parameters:
    - path (str): Path to the CSV file.
    - names (list[str]): List of column names for the CSV file.
    - dtype (dict): Dictionary specifying the data types for columns in the DataFrame.
    - text_cols (list[str]): List of column names containing text data in the dataset.
    - tokenizer (AutoTokenizer): The tokenizer used for tokenizing text data.
    - categorical_cols (list[str]): List of column names containing categorical variables in the dataset.
    - numerical_cols (list[str]): List of column names containing numerical variables in the dataset.
    - categorical_encode_type (str): Type of encoding for categorical variables (e.g., 'one-hot', 'label').
    - label_col (str): Name of the column containing labels in the dataset.
    - label_list (list[int]): List of possible label values.

    Returns:
    - dataset (torch.utils.data.dataset): A dataset consistent with pytorch requirements

    Usage Example:
    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    path = "path/to/dataset.csv"
    names = ["id", "leiden_90", "louvain_90", "text_col", "label_col"]
    dtype = {"id": int, "leiden_90": int, "louvain_90": int, "text_col": str, "label_col": int}
    text_cols = ["text_col"]
    categorical_cols = ["leiden_90", "louvain_90"]
    numerical_cols = []
    categorical_encode_type = "label"
    label_col = "label_col"
    label_list = [0, 1]

    dataset = read_and_convert_df(
        path,
        names,
        dtype,
        text_cols,
        tokenizer,
        categorical_cols,
        numerical_cols,
        categorical_encode_type,
        label_col,
        label_list,
    )
    """
    df= pd.read_csv(
        path,
        names=names,
        #dtype=dtype,
        header=0,
        na_values=["", "[]"],
        lineterminator="\n",
    )
    df=df[df["leiden_90"].notna()]
    df=df[df["louvain_90"].notna()]
    df.set_index("id")
    dataset=load_data(df,
        text_cols=text_cols,
        tokenizer=tokenizer,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        categorical_encode_type=categorical_encode_type,
        label_col=label_col,
        label_list=label_list,
        sep_text_token_str=tokenizer.sep_token,
    )
    return dataset

def set_tab_config(
    combine_feat_method: str,
    cat_feat_dim: int,
    numerical_feat_dim: int,
    num_labels: int
) -> TabularConfig:
    """
    Set configuration parameters for tabular data processing.

    Parameters:
    - combine_feat_method (str): Method of combining tabular data.
    - cat_feat_dim (int): Dimensionality of categorical features.
    - numerical_feat_dim (int): Dimensionality of numerical features.
    - num_labels (int): Number of labels in the classification task.

    Returns:
    - tabular_config (TabularConfig): Configuration object for tabular data processing.

    Usage Example:
    ```python
    combine_feat_method = "method_name"
    cat_feat_dim = 10
    numerical_feat_dim = 5
    num_labels = 2

    tabular_config = set_tab_config(
        combine_feat_method,
        cat_feat_dim,
        numerical_feat_dim,
        num_labels
    )
    ```

    Note:
    - `combine_feat_method`: Specify the method of combining tabular data the choice is between:
        'concat',
        'mlp_on_categorical_then_concat',
        'individual_mlps_on_cat_and_numerical_feats_then_concat',
        'mlp_on_concatenated_cat_and_numerical_feats_then_concat',
        'attention_on_cat_and_numerical_feats',
        'gating_on_cat_and_num_feats_then_sum',
    """
    tabular_config = TabularConfig(
            combine_feat_method=combine_feat_method,  # change this to specify the method of combining tabular data
            cat_feat_dim=cat_feat_dim,  # need to specify this
            numerical_feat_dim=numerical_feat_dim,  # need to specify this
            num_labels=num_labels,   # need to specify this, assuming our task is binary classification
    )
    return tabular_config

def set_training_args(
    overwrite_output_dir: bool,
    do_train: bool,
    do_eval: bool,
    per_device_train_batch_size: int,
    num_train_epochs: int,
    logging_steps: int,
    eval_steps: int,
    auto_find_batch_size: bool,
    dataloader_drop_last: bool
) -> TrainingArguments:
    """
    Set training configuration parameters.

    Parameters:
    - overwrite_output_dir (bool): Whether to overwrite the output directory if it exists.
    - do_train (bool): Whether to perform training.
    - do_eval (bool): Whether to perform evaluation.
    - per_device_train_batch_size (int): Batch size per CPU for training.
    - num_train_epochs (int): Number of training epochs.
    - logging_steps (int): Number of steps between each logging update.
    - eval_steps (int): Number of steps between each evaluation run.
    - auto_find_batch_size (bool): Whether to automatically find an efficient batch size.
    - dataloader_drop_last (bool): Whether to drop the last batch in dataloader if its size is smaller than the specified batch size.

    Returns:
    - training_args (TrainingArguments): Configuration object for training.
    """
    training_args = TrainingArguments(
        output_dir=TRANSFORMERS_CACHE_DIR,
        logging_dir=TRANSFORMERS_CACHE_DIR,
        overwrite_output_dir=overwrite_output_dir,
        do_train=do_train,
        do_eval=do_eval,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        weight_decay=weight_decay,
        auto_find_batch_size=auto_find_batch_size,
        dataloader_drop_last=dataloader_drop_last
        )
    return training_args
def main():
    """Do the main"""
    tokenizer = AutoTokenizer.from_pretrained(bert)
    train_dataset=read_and_convert_df(training_path[0],
                                      names_dataset,
                                      dtype_dataset,
                                      text_cols,
                                      tokenizer,
                                      categorical_cols,
                                      numerical_cols,
                                      categorical_encode_type,
                                      label_col,
                                      label_list)
    val_dataset=read_and_convert_df(training_path[1],
                                      names_dataset,
                                      dtype_dataset,
                                      text_cols,
                                      tokenizer,
                                      categorical_cols,
                                      numerical_cols,
                                      categorical_encode_type,
                                      label_col,
                                      label_list)
    test_dataset=read_and_convert_df(training_path[2],
                                      names_dataset,
                                      dtype_dataset,
                                      text_cols,
                                      tokenizer,
                                      categorical_cols,
                                      numerical_cols,
                                      categorical_encode_type,
                                      label_col,
                                      label_list)
    tabular_config=set_tab_config(combine_feat_method,cat_feat_dim,numerical_feat_dim,len(label_list))
    hf_config = AutoConfig.from_pretrained(bert)
    hf_config.tabular_config = tabular_config
    model = AutoModelWithTabular.from_pretrained(bert, config=hf_config)
    training_args=set_training_args(overwrite_output_dir,do_train,do_eval,per_device_train_batch_size,num_train_epochs,logging_steps,eval_steps,auto_find_batch_size,dataloader_drop_last)
    set_seed(training_args.seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calc_classification_metrics
    )
    trainer.train()
    trainer.evaluate(eval_dataset=val_dataset)
    
if __name__ == "__main__":
    os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
    from configuration_params import bert,dataset_params,tab_conf_params,training_args_params,label_list
    training_path,names_dataset,dtype_dataset,categorical_cols,numerical_cols,text_cols,categorical_encode_type,label_col,label_list=dataset_params
    combine_feat_method,cat_feat_dim,numerical_feat_dim=tab_conf_params
    overwrite_output_dir,do_train,do_eval,per_device_train_batch_size,num_train_epochs,logging_steps,eval_steps,weight_decay,auto_find_batch_size,dataloader_drop_last=training_args_params
    main()
    
