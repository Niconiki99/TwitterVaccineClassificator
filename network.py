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
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR, NETWORK_DAT
from fa2 import ForceAtlas2
MAKE=True #Set it true to make again the positions of the users, using fa2
def col(utoplot,com2col,mappa,df):
    colors=np.empty(len(utoplot),dtype="<U20")
    j=0
    for i,uid in enumerate(utoplot):
        try:
            colors[i]=com2col[mappa[uid]]
        except KeyError:
            colors[i]=com2col[7]
            j+=1
    return colors,j

df_com = pd.read_csv(
      DATA_DIR+"communities_2021-06-01.csv.gz",
        lineterminator="\n"
    )


df_com=df_com[['Unnamed: 0','leiden_90', 'louvain_90']].rename(columns={'Unnamed: 0':'user.id','leiden_90':'leiden_90','louvain_90':'louvain_90'})
#df_com.set_index("user.id")
df_com["user.id"]=df_com["user.id"].apply(str)
dict=df_com.set_index("user.id").to_dict("series")
leiden=dict["leiden_90"]
louvain=dict["louvain_90"]
G=nx.read_graphml(NETWORK_DATA+"retweet_graph_undirected_2021-06-01.graphml")
if MAKE:
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
    with open(DATA_DIR+'position.json', 'w') as convert_file: 
     convert_file.write(json.dumps(positions))
else:
    with open(DATA_DIR+'position.json') as f: 
        positions = json.load(f)

#DRAWING
utoplot=list(positions.keys())
sizes=np.array([G.degree(u) for u in utoplot])
sizes=np.interp(sizes, (sizes.min(), sizes.max()), (5, 500))
com_edges=list(G.edges())
print(len(com_edges), 'edges in the network')

xfa=[positions[u][0] for u in utoplot]
yfa=[positions[u][1] for u in utoplot]
#com_ids={c:list(set([u for u,com in com_of_user.items() if com==c])) for c in comtodraw}

#com2col={'A':'dodgerblue','B':'coral','C':'orchid','D':'deepskyblue','E':'orange',
#        'F':'grey','G':'deeppink','H':'black','I':'yellow','J':'brown','K':'cyan',
#        'L':'lime','M':'green','N':'slateblue'}
com2col=['dodgerblue','orange','grey','cyan','pink','peru','white','rebeccapurple']
#com2col={com:(col  if com=='I' else 'lightgrey') for com,col in com2col.items()}
colors_ld,non_com_ld=col(utoplot,com2col,leiden,df_com)
colors_lv,non_com_lv=col(utoplot,com2col,louvain,df_com)
print(str(non_com_ld)+" elements are in no comunity with leiden method")
print(str(non_com_lv)+" elements are in no comunity with louvain method")
#colors=[com2col[jo] if com_of_user[u]==jo else 'lightgrey' for u in Gcomp.nodes]
#colors=['red' if com_of_user[u]==jo else 'lightgrey' for u in Gcomp.nodes]
#color="red"
s_t=time.time()
f=plt.figure(dpi=500,figsize=(12,12))
plt.scatter(xfa,yfa,c=colors_ld,s=sizes,alpha=.6,
            marker='.',edgecolors='black',linewidths=0)
#nx.draw_networkx_edges(
#    G, positions,edgelist=com_edges,alpha=0.01,edge_color='white',
#    connectionstyle="arc3,rad=0.3"  # <-- THIS IS IT
#)
#markers = [plt.Line2D([0,0],[0,0],color=colors, marker='o', linestyle='') for com,color in com2col.items() if com!='XX']
#scritte=[c+': {:.1f}%'.format(100*com2size[c]) for c in comtodraw]
#plt.legend(markers, scritte, numpoints=1,loc='lower left',#'upper right',
#          fontsize=14)3
#plt.xlim(-6000,7000)
#plt.ylim(-6000,6000)

plt.axis('off')
plt.gca().set_facecolor('black')
f.set_facecolor('black')


plt.savefig(DATA_DIR+'net_ld.png', bbox_inches='tight')
#plt.savefig('images/'+filen+'_j'+jo+'.png', bbox_inches='tight')
plt.clf()
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))
s_t=time.time()
f=plt.figure(dpi=500,figsize=(12,12))
plt.scatter(xfa,yfa,c=colors_lv,s=sizes,alpha=.6,
            marker='.',edgecolors='black',linewidths=0)
#nx.draw_networkx_edges(
#    G, positions,edgelist=com_edges,alpha=0.01,edge_color='white',
#    connectionstyle="arc3,rad=0.3"  # <-- THIS IS IT
#)
#markers = [plt.Line2D([0,0],[0,0],color=colors, marker='o', linestyle='') for com,color in com2col.items() if com!='XX']
#scritte=[c+': {:.1f}%'.format(100*com2size[c]) for c in comtodraw]
#plt.legend(markers, scritte, numpoints=1,loc='lower left',#'upper right',
#          fontsize=14)3
#plt.xlim(-6000,7000)
#plt.ylim(-6000,6000)

plt.axis('off')
plt.gca().set_facecolor('black')
f.set_facecolor('black')


plt.savefig(DATA_DIR+'net_lv.png', bbox_inches='tight')
#plt.savefig('images/'+filen+'_j'+jo+'.png', bbox_inches='tight')
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))