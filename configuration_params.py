import pathlib
#DIRECTORIES
from DIRS import TRANSFORMERS_CACHE_DIR,DATA_DIR,LARGE_DATA_DIR,NETWORK_DATA
#BuildGraph params
DATAPATH = pathlib.Path(DATA_DIR) #path storing the datas
deadline="2021-06-01" #deadline selected for the parsing
#BuildCom params
NETPATH = pathlib.Path(NETWORK_DATA) #path storing the networks
#Preprocess params
#LIST OF PARAMETERS FOR THE READING OF THE DATAFRAMES
path_df=LARGE_DATA_DIR+"df_full.csv.gz"
name_df=['id',
         'created_at', 
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
         'lang']
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
           "louvain":int,
           "leiden_5000":int,
           "leiden_90":int,
           "louvain_5000":int,
           "louvain_90":int,}
names_com=["user.id",
           "leiden",
           "louvain",
           "leiden_5000",
           "leiden_90",
           "louvain_5000",
           "louvain_90"]
path_pos=DATA_DIR+'position.json'
names_pos=["x_pos","y_pos"]
labels=['ProVax','AntiVax','Neutral']#labels necessary for machine learning
random_state=42 #random state
#network params
MAKE=True #Set it true to make again the positions of the users, using fa2
com2col=['dodgerblue','orange','grey','cyan','pink','peru','white','rebeccapurple'] #list of colors used for the drawing
path_to_save=DATA_DIR+'position.json'
path_to_read=DATA_DIR+'position.json'
###########################################
#Machine learning parameters
bert='m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0'
#DATASET PARAMS
train_path=DATA_DIR+"train.csv"
val_path=DATA_DIR+"val.csv"
test_path=DATA_DIR+"test.csv"
training_path=[train_path,val_path,test_path]
names_dataset=[
            "id",
            "sentence",
            "label",
            "leiden_90",
            "louvain_90",
            "x_pos",
            "y_pos"
]
dtype_dataset={
            "id": str,
            "sentence": str,
            "label": int,
            "leiden_90":int,
            "louvain_90":int,
            "x_pos":float,
            "y_pos":float,            
        }
text_cols=["sentence"]
categorical_cols=["leiden_90","louvain_90"]
numerical_cols=["x_pos","y_pos"]
categorical_encode_type="ohe"
numerical_transformer_method='none'
label_col="label"
label_list=[0,1,2]
dataset_params=(training_path,names_dataset,dtype_dataset,categorical_cols,numerical_cols,text_cols,categorical_encode_type,label_col,label_list)
#tabular config params
combine_feat_method='text_only'
cat_feat_dim=0
numerical_feat_dim=0
tab_conf_params=(combine_feat_method,cat_feat_dim,numerical_feat_dim)
#training args params
use_cpu =True
overwrite_output_dir=True
do_train=True
do_eval=True
per_device_train_batch_size=32
num_train_epochs=3
logging_steps=25
eval_steps=250
weight_decay=float(0.01),
auto_find_batch_size=True,
dataloader_drop_last=True
training_args_params=(overwrite_output_dir,do_train,do_eval,per_device_train_batch_size,num_train_epochs,logging_steps,eval_steps,weight_decay,auto_find_batch_size,dataloader_drop_last)