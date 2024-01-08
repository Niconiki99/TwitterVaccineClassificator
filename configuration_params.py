from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR,NETWORK_DATA
import pathlib
#BuildGraph params
DATAPATH = pathlib.Path(DATA_DIR) #path storing the datas
deadline="2021-06-01" #deadline selected for the parsing
#BuildCom params
NETPATH = pathlib.Path(NETWORK_DATA) #path storing the networks
#Preprocess params
#LIST OF PARAMETERS FOR THE READING OF THE DATAFRAMES
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
labels=['ProVax','AntiVax','Neutral']#labels necessary for machine learning
random_state=42 #random state
#network params
MAKE=True #Set it true to make again the positions of the users, using fa2
com2col=['dodgerblue','orange','grey','cyan','pink','peru','white','rebeccapurple'] #list of colors used for the drawing
path_to_save=DATA_DIR+'position.json'
path_to_read=DATA_DIR+'position.json'