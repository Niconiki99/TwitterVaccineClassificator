import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR

def undersampling(df, random_state):
    # undersampling function, since label classes are unbalancd (ProVax is the smallest)
    l = df.label.value_counts()['ProVax']
    out = pd.concat([
        df[df.label == 'ProVax'],
        df[df.label == 'AntiVax'].sample(l, random_state=random_state),
        df[df.label == 'Neutral'].sample(l, random_state=random_state)
    ], ignore_index=False)
    return out
#READING OF THE DATAS
#DATASET WITH ALL THE TWEETS
df = pd.read_csv(
      LARGE_DATA_DIR+"df_full.csv.gz",
        index_col="id",
        dtype={
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
        },
        na_values=["", "[]"],
        parse_dates=["created_at"],
        lineterminator="\n",
    )
#DATASET WITH THE COMUNITIES
df_com = pd.read_csv(
    DATA_DIR+"communities_2021-06-01.csv.gz",
    lineterminator="\n",
    header=0,
    names=["user.id","leiden","louvain","leiden_5000","leiden_90","louvain_5000","louvain_90"],
    dtype={"user.id":str,
           "leiden":int,
           "louvain":int,
           "leiden_5000":int,
           "leiden_90":int,
           "louvain_5000":int,
           "louvain_90":int,}
    )
df_com=df_com[["user.id","leiden_90","louvain_90"]]

#DATASET WITH THE POSITIONS
with open(DATA_DIR+'position.json') as f: 
    positions = json.load(f)     
df_pos=pd.DataFrame.from_dict(positions, orient='index',columns=["x_pos","y_pos"])

#MERGING OF THE THREE DATAFRAMES
df=df.merge(df_com,how="left",left_on="user.id",right_on="user.id")
df=df.merge(df_pos,how="left",left_on="user.id",right_index=True)

#CLEANUP OF THE DATAFRAME'S TEXT
df_anno=df[df['annotation'].notna()]
df_anno.loc[:,'text']=df_anno['text'].apply(lambda x: x.replace('\n',' ').replace('\t','').replace("\r\n"," ").replace('\u0085'," ").replace('\u2028'," ").replace('\u2029'," "))

#RENAMING AND REMOVING COLUMNS
df_anno=df_anno[['text','annotation',"leiden_90","louvain_90","x_pos","y_pos"]].rename(columns={'text':'sentence','annotation':'label',"leiden_90":"leiden_90","louvain_90":"louvain_90","x_pos":"x_pos","y_pos":"y_pos"})
df_anno["leiden_90"]=df_anno["leiden_90"].apply(int)
df_anno["louvain_90"]=df_anno["louvain_90"].apply(int)

#TRAIN TEST SPLITTING WITH THREE LABELS
df_out=undersampling(df_anno,42)
id_train,id_test=train_test_split(df_out.index, test_size=0.33, random_state=42)
id_test,id_val=train_test_split(id_test, test_size=0.5, random_state=42)
id2label = {0:'AntiVax', 1:'Neutral', 2:'ProVax'}
label2id = {'AntiVax':0, 'Neutral':1, 'ProVax':2}
df_out["label"]=df_out["label"].map(label2id).dropna()
#PRINTING OF THE DATAFRAMES
df_out.loc[id_train].to_csv(DATA_DIR+'train.csv',lineterminator='\n')
df_out.loc[id_test].to_csv(DATA_DIR+'test.csv',lineterminator='\n')
df_out.loc[id_val].to_csv(DATA_DIR+'val.csv',lineterminator='\n')
df_out.to_csv(DATA_DIR+'sampled.csv',lineterminator='\n')
df_anno.to_csv(DATA_DIR+'annotated.csv',lineterminator='\n')

#RELABELING FOR 2 LABEL SITUATION
id2label = {0:'AntiVax', 1:'Neutral', 2:'ProVax'}
label2id = {'AntiVax':0, 'Neutral':np.nan,'ProVax':1}
df_red=df_out.copy(deep=True)
df_red["label"]=df_out["label"].map(id2label).map(label2id).dropna()
df_red=df_red.dropna()
df_red["label"]=df_red["label"].apply(int)
#PRINTING OF THE DATAFRAMES
df_red[df_red.index.isin(id_train)].to_csv(DATA_DIR+'train_2l.csv',lineterminator='\n')
df_red[df_red.index.isin(id_test)].to_csv(DATA_DIR+'test_2l.csv',lineterminator='\n')
df_red[df_red.index.isin(id_val)].to_csv(DATA_DIR+'val_2l.csv',lineterminator='\n')