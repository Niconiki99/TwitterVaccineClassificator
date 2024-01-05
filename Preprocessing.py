import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
labels=['ProVax','AntiVax','Neutral']
random_state=42

def undersampling(df, random_state):
    # undersampling function, since label classes are unbalancd (ProVax is the smallest)
    l = df.label.value_counts()['ProVax']
    out = pd.concat([
        df[df.label == 'ProVax'],
        df[df.label == 'AntiVax'].sample(l, random_state=random_state),
        df[df.label == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=False)
    return out

def reading_merging(path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,name_pos):
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
        names=names_com,
        dtype=dtype_df,
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

def preproc(df,label,seed):
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

def main(DATA_INFO):
    path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,name_pos,seed,label,DATA_PATH=DATA_INFO
    df,ids=preproc(reading_merging(path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,name_pos,dtype_pos),label,seed)
    id_train,id_test=train_test_split(ids, test_size=0.33, random_state=42)
    id_test,id_val=train_test_split(ids, test_size=0.5, random_state=42)
    df[df.index.isin(id_train)].to_csv(DATA_PATH+'train.csv',lineterminator='\n')
    df[df.index.isin(id_test)].to_csv(DATA_PATH+'test.csv',lineterminator='\n')
    df[df.index.isin(id_val)].to_csv(DATA_PATH+'val.csv',lineterminator='\n')
    

if __name__ == "__main__":
    path_df=LARGE_DATA_DIR+"df_full.csv.gz"
    name_df=['created_at', 
             'text', 
             'user.id',              
             'user.screen_name', 
             'place', 
             'url',       
             'retweeted_status.id', 
             'retweeted_status.user.id',       
             'retweeted_status.url', 
             'annotation', 
             'user_annotation', 
             'lang',       
             'leiden_90', 
             'louvain_90', 
             'x_pos', 
             'y_pos']
    dtype_df={
            "id": str,
            "text": str,
            "user.id": str,
            "user.screen_name": str,
            "place": str,
            "url": str,
            "retweeted_status.id": str,
            "retweeted_status.user.id": str,
            "retweeted_status.url": str,
            "annotation": str,
            "user_annotation": str,
            "lang": str,
        }
    path_com=DATA_DIR+"communities_2021-06-01.csv.gz"
    dtype_com={"user.id":str,
           "leiden":int,
           "infomap":int,
           "louvain":int,
           "leiden_5000":int,
           "leiden_90":int,
           "louvain_5000":int,
           "louvain_90":int,
           "infomap_5000":int,
           "infomap_90":int}
    names_com=["user.id","leiden","infomap","louvain","leiden_5000","leiden_90","louvain_5000","louvain_90","infomap_5000","infomap_90"]
    path_pos=DATA_DIR+'position.json'
    names_pos=["x_pos","y_pos"]
    DATA_INFO=(path_df,name_df,dtype_df,path_com,names_com,dtype_com,path_pos,names_pos,random_state,labels,DATA_DIR)
    main(DATA_INFO)