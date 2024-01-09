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
import seaborn as sns
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
def read_dataset(tokenizer,data_path,text_cols,categorical_encode_type,numerical_transformer_method,label_col,label_list):
    train_dataset, val_dataset, test_dataset = load_data_from_folder(
        data_path,
        text_cols=text_cols,
        tokenizer=tokenizer,
        categorical_encode_type=categorical_encode_type,
        numerical_transformer_method=numerical_transformer_method,
        label_col=label_col,
        label_list=label_list,
        sep_text_token_str=tokenizer.sep_token,
    )
    return train_dataset,val_dataset,test_dataset
    
def read_and_convert_df(path,names,dtype,text_cols,tokenizer,categorical_cols,numerical_cols,categorical_encode_type,label_col,label_list):
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

def set_tab_config(combine_feat_method,cat_feat_dim,numerical_feat_dim,num_labels):
    tabular_config = TabularConfig(
            combine_feat_method=combine_feat_method,  # change this to specify the method of combining tabular data
            cat_feat_dim=cat_feat_dim,  # need to specify this
            numerical_feat_dim=numerical_feat_dim,  # need to specify this
            num_labels=num_labels,   # need to specify this, assuming our task is binary classification
    )
    return tabular_config

def set_training_args(overwrite_output_dir,do_train,do_eval,per_device_train_batch_size,num_train_epochs,logging_steps,eval_steps,auto_find_batch_size,dataloader_drop_last):
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
    
