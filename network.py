import networkx as nx
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re,os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configuration_params import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR, NETWORK_DATA
from fa2 import ForceAtlas2

def load_com(path_com,names_com,dtype_com):
    df_com = pd.read_csv(
        path_com,
        header=0,
        names=["user.id","leiden","louvain","leiden_5000","leiden_90","louvain_5000","louvain_90"],
        dtype=dtype_com,
        lineterminator="\n"
    )
    dict=df_com.set_index("user.id").to_dict("series")
    leiden=dict["leiden_90"]
    louvain=dict["louvain_90"]
    return(leiden,louvain)

def position_creation(G,path_to_save=DATA_DIR+'position.json'):
    forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    with open(path_to_save, 'w') as convert_file: 
        convert_file.write(json.dumps(positions))
    return positions

def pos_reading(path_to_read=DATA_DIR+'position.json'):
    with open(DATA_DIR+'position.json') as f: 
        positions = json.load(f)
    return positions

def drawing_params(positions,G,com2col,mappa):
    utoplot=list(positions.keys())
    sizes=np.array([G.degree(u) for u in utoplot])
    sizes=np.interp(sizes, (sizes.min(), sizes.max()), (5, 500))
    com_edges=list(G.edges())
    print(len(com_edges), 'edges in the network')
    xfa=[positions[u][0] for u in utoplot]
    yfa=[positions[u][1] for u in utoplot]
    colors=np.empty(len(utoplot),dtype="<U20")
    for i,uid in enumerate(utoplot):
        try:
            colors[i]=com2col[mappa[uid]]
        except (KeyError,IndexError):
            colors[i]=com2col[7]
    return (utoplot,sizes,com_edges,xfa,yfa,colors)
    
def drawing(drawing_params,path_to_draw=DATA_DIR+"network_map.pdf"):
    utoplot,sizes,com_edges,xfa,yfa,colors=drawing_params
    s_t=time.time()
    f=plt.figure(dpi=500,figsize=(12,12))
    plt.scatter(xfa,yfa,c=colors,s=sizes,alpha=.6,
                marker='.',edgecolors='black',linewidths=0)
    plt.axis('off')
    plt.gca().set_facecolor('black')
    f.set_facecolor('black')
    plt.savefig(path_to_draw, bbox_inches='tight')
    plt.close()
    e_t=time.time()-s_t
    print('Elapsed time: {} min'.format(e_t/60))
    
def main(MAKE,com2col,path_com,dtype_com,names_com,path_to_save=DATA_DIR+'position.json',path_to_read=DATA_DIR+'position.json'):
    G=nx.read_graphml(NETWORK_DATA+"/retweet_graph_undirected_2021-06-01.graphml")
    leiden,louvain=load_com(path_com,names_com,dtype_com)
    if MAKE:
        positions=position_creation(G)
    else:
        positions=pos_reading()
    drawing(drawing_params(positions,G,com2col,leiden))


if __name__ == "__main__":
    from configuration_params import MAKE,com2col,path_com,dtype_com,names_com
    main(MAKE,com2col,path_com,dtype_com,names_com)

