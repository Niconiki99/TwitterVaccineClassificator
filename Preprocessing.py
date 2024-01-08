"""
The program is designed for preprocessing a dataset and preparing it for machine learning tasks. 
It involves reading data from multiple sources, merging them into a single DataFrame, performing undersampling to balance label classes, and splitting the data into training, testing, and validation sets. The resulting datasets are then saved as CSV files.
Data Loading and Merging:
    Read the main dataset, communities dataset, and positions dataset.
    Merge these datasets into a single DataFrame.
Preprocessing:
    Perform text cleaning and preprocessing on relevant columns.
    Balance label classes using undersampling.
    Map labels to numerical values.
Train-Test-Validation Split:
    Split the preprocessed data into training, testing, and validation sets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from configuration_params import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR,labels,random_state

def undersampling(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """
    Perform undersampling on a DataFrame to balance label classes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'label' column.
    - random_state (int): Seed for reproducibility in random sampling.

    Returns:
    - pd.DataFrame: Undersampled DataFrame with balanced label classes.

    """
    l = df.label.value_counts()['ProVax']
    out = pd.concat([
        df[df.label == 'ProVax'],
        df[df.label == 'AntiVax'].sample(l, random_state=random_state),
        df[df.label == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=False)
    return out

def reading_merging(path_df: str,
    name_df: list,
    dtype_df: dict,
    path_com: str,
    names_com: list,
    dtype_com: dict,
    path_pos: str,
    name_pos: list
) -> pd.DataFrame:
    """
    Read and merge data from multiple sources into a single DataFrame.

    Parameters:
    - path_df (str): Filepath for the main DataFrame CSV file.
    - name_df (list): Column names for the main DataFrame.
    - dtype_df (dict): Data types for columns in the main DataFrame.
    - path_com (str): Filepath for the community DataFrame CSV file.
    - names_com (list): Column names for the community DataFrame.
    - dtype_com (dict): Data types for columns in the community DataFrame.
    - path_pos (str): Filepath for the JSON file containing position data.
    - name_pos (list): Column names for the position DataFrame.

    Returns:
    - pd.DataFrame: Merged DataFrame containing data from the main DataFrame, community DataFrame, and position data.
    """
    df = pd.read_csv(
        path_df,
        index_col="id",
        name=name_df,
        dtype=dtype_df,
        na_values=["", "[]"],
        parse_dates=["created_at"],
        lineterminator="\n",
        )
    df_com = pd.read_csv(
        path_com,
        header=0,
        names=names_com,
        dtype=dtype_com,
        lineterminator="\n"
    )
    df_com=df_com[["user.id","leiden_90","louvain_90"]]
    df_com["user.id"]
    with open(path_pos) as f: 
        positions = json.load(f)     
    # reconstructing the data as a dictionary 
    df_pos=pd.DataFrame.from_dict(positions, orient='index',columns=name_pos)
    df=df.merge(df_com,how="left",left_on="user.id",right_on="user.id")
    df=df.merge(df_pos,how="left",left_on="user.id",right_index=True)
    return df

def preproc(df: pd.DataFrame,
    label: list,
    seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess DataFrame for text classification.
    Missing values in the text will be replaced, unused columns will be removed, and used columns will be renamed.
    Type checking for the columns (e.g leiden_90 and louvain_90 are fixed to int).
    Label column are mapped to int.
    Undersampling is applied.
    

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing text data and annotations.
    - label (list): List of labels for classification.
    - seed (int): Random seed for undersampling.

    Returns:
    - Tuple[pd.DataFrame, np.ndarray]: Tuple containing the preprocessed DataFrame and corresponding indices.
    """
    df_anno=df[df['annotation'].notna()]
    df_anno.loc[:,'text']=df_anno['text'].apply(lambda x: x.replace('\n',' ').replace('\t','').replace("\r\n"," ").replace('\u0085'," ").replace('\u2028'," ").replace('\u2029'," "))
    df_anno=df_anno[['text','annotation',"leiden_90","louvain_90","x_pos","y_pos"]].rename(columns=
                                                                                               {'text':'sentence',
                                                                                                'annotation':'label',
                                                                                                "leiden_90":"leiden_90",
                                                                                                "louvain_90":"louvain_90",
                                                                                                "x_pos":"x_pos",
                                                                                                "y_pos":"y_pos"})
    df_anno["leiden_90"]=df_anno["leiden_90"].apply(int)
    df_anno["louvain_90"]=df_anno["louvain_90"].apply(int)
    label2id = {label[0]:0, label[1]:1, label[2]:2}
    df_anno=undersampling(df_anno,seed)
    ids=df_anno.index.to_numpy()
    if(len(label)==2):
        label2id = {label[0]:0, label[1]:np.nan, label[2]:1}
    df_anno["label"]=df_anno["label"].map(label2id).dropna()
    df_anno["label"]=df_anno["label"].apply(int)
    return (df_anno,ids)

def main(DATA_INFO: Tuple):
    """Do the main"""
    path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,name_pos,seed,label,DATA_PATH=DATA_INFO
    df,ids=preproc(reading_merging(path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,name_pos,dtype_pos),label,seed)
    id_train,id_test=train_test_split(ids, test_size=0.33, random_state=42)
    id_test,id_val=train_test_split(ids, test_size=0.5, random_state=42)
    df[df.index.isin(id_train)].to_csv(DATA_PATH+'train.csv',lineterminator='\n')
    df[df.index.isin(id_test)].to_csv(DATA_PATH+'test.csv',lineterminator='\n')
    df[df.index.isin(id_val)].to_csv(DATA_PATH+'val.csv',lineterminator='\n')
    

if __name__ == "__main__":
    from configuration_params import path_df,name_df,dtype_df,path_com,dtype_com,names_com,path_pos,names_pos
    DATA_INFO=(path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,names_pos,random_state,labels,DATA_DIR)
    main(DATA_INFO)