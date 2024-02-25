"""
This code is designed for analyzing and visualizing communities within a retweet network, and to build a dictionary of positions, which is used as a feature in classification problem.
Position Generation: The script employs the ForceAtlas2 layout algorithm to generate node positions for a given graph. The position_creation function takes a NetworkX graph and produces positions that are then saved in JSON format.
Drawing Parameters: A function (drawing_params) is implemented to compute various parameters for drawing a graph, such as user IDs, node sizes, edges, coordinates, and colors based on community assignments.
Drawing: The script utilizes the generated parameters to draw a network map using Matplotlib. The drawing function creates a scatter plot with customizable features, including node sizes, colors, and background.
Configuration Parameters:The configuration parameters include flags (MAKE), community mapping (com2col), file paths (path_com, path_to_save, path_to_read), and data types (dtype_com, names_com). These parameters can be adjusted to tailor the analysis to different datasets or requirements.
Note:
Ensure that the required libraries are installed before running the script.
The script assumes the existence of a retweet network graph stored in the specified graphml file (retweet_graph_undirected_2021-06-01.graphml). Adjust the file path if needed.
 """
import networkx as nx
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import re,os
import pandas as pd
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2
from configobj import ConfigObj

def load_com(path_com: str, names_com: list, dtype_com: dict) -> tuple[pd.Series, pd.Series]:
    """
    Load community data from a CSV file and extract specific community assignments.
    It extracts the leiden_90 and louvain_90 communities, which means the 90% coverage reduced communities.

    Parameters:
    - path_com (str): Path to the CSV file containing community data.
    - names_com (list): list of column names in the community data.
    - dtype_com (dict): Dictionary specifying data types for each column.

    Returns:
    - tuple[pd.Series, pd.Series]: tuple containing community assignments (leiden, louvain).
    """
    df_com = pd.read_csv(
        path_com,
        header=0,
        names=names_com,
        dtype=dtype_com,
        lineterminator="\n"
    )
    dict=df_com.set_index("user.id").to_dict("series")
    leiden=dict["leiden_90"]
    louvain=dict["louvain_90"]
    return(leiden,louvain)

def position_creation(G: nx.Graph, path_to_save: str = 'position.json') -> dict:
    """
    Generate and save node positions using ForceAtlas2 layout for a given graph.

    Parameters:
    - G (nx.Graph): NetworkX graph for which positions are to be generated.
    - path_to_save (str): Path to save the generated positions in JSON format (default is DATA_DIR + 'position.json').

    Returns:
    - dict: Dictionary containing node positions.

    Notes:
    - The function uses the ForceAtlas2 layout algorithm to generate node positions for the given graph.
    - The positions are saved in JSON format at the specified path.
    """
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

def pos_reading(path_to_read: str = 'position.json') -> dict:
    """
    Read node positions from a JSON file.

    Parameters:
    - path_to_read (str): Path to the JSON file containing node positions (default is 'position.json').

    Returns:
    - dict: Dictionary containing node positions.
    """
    with open(path_to_read) as f: 
        positions = json.load(f)
    return positions

def drawing_params(
    positions: dict, G: nx.Graph, com2col: list, mappa: pd.Series
) -> tuple[list, np.ndarray, list, list, list, np.ndarray]:
    """
    Generate parameters for drawing a graph.
    Lists the users to be plotted.
    Computes the size of the points depending on the degree, and computes the number of edges.
    Unpacks the positions in x and y coordinates.
    Couple each point with a color, depending on the community.

    Parameters:
    - positions (dict): Dictionary containing node positions.
    - G (nx.Graph): NetworkX graph object.
    - com2col (dict): Dictionary mapping community IDs to colors.
    - mappa (dict): Dictionary mapping node IDs to community IDs.

    Returns:
    - tuple[list, np.ndarray, list, list, list, np.ndarray]:
        - list: list of users IDs to be plotted.
        - np.ndarray: Array of node sizes based on degree.
        - list: list of edges in the network.
        - list: list of x-coordinates for nodes.
        - list: list of y-coordinates for nodes.
        - np.ndarray: Array of colors for nodes.
    """
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
    
def drawing(
    drawing_params: tuple[list, np.ndarray, list, list, list, np.ndarray],
    path_to_draw: str = "network_map.pdf"
) -> None:
    """
    Draw a network map using specified parameters.
    The map is drawn as a scatter plot.

    Parameters:
    - drawing_params (tuple[list, np.ndarray, list, list, list, np.ndarray]):
        tuple containing parameters for drawing a graph:
            - list: list of users IDs to be plotted.
            - np.ndarray: Array of node sizes based on degree.
            - list: list of edges in the network.
            - list: list of x-coordinates for nodes.
            - list: list of y-coordinates for nodes.
            - np.ndarray: Array of colors for nodes.
    - path_to_draw (str): Filepath to save the network map (default is "network_map.pdf").

    Returns:
    - None
    """
    utoplot,sizes,com_edges,xfa,yfa,colors=drawing_params
    s_t=time.time()
    f=plt.figure(dpi=500,figsize=(12,12))
    plt.scatter(xfa,yfa,c=tuple(colors),s=sizes,alpha=.6,
                marker='.',edgecolors='black',linewidths=0)
    plt.axis('off')
    plt.gca().set_facecolor('black')
    f.set_facecolor('black')
    plt.savefig(path_to_draw, bbox_inches='tight')
    plt.close()
    e_t=time.time()-s_t
    print('Elapsed time: {} min'.format(e_t/60))
    
def main() -> None:
    """
    Do the main
    """
    #importing the parameters from config.txt
    config= ConfigObj("config.txt")
    path=config["READING_PARAMS"]["DF_FULL"]["path"]
    TRANSFORMERS_CACHE_DIR=config["DIRS"]["TRANSFORMERS_CACHE_DIR"]
    LARGE_DATA_DIR=config["DIRS"]["LARGE_DATA_DIR"]
    NETWORK_DATA=config["DIRS"]["NETWORK_DATA"]
    DATA_DIR=config["DIRS"]["DATA_DIR"]
    path_com=config["READING_PARAMS"]["DF_COM"]["path"]
    MAKE=config["NETWORK_PARAMS"].as_bool("MAKE")
    com2col=config["NETWORK_PARAMS"].as_list("com2col")
    dtype_com=config["READING_PARAMS"]["DF_COM"]["dtype_df"]
    names_com=config["READING_PARAMS"]["DF_COM"].as_list("col_name")
    G=nx.read_graphml(NETWORK_DATA+"/retweet_graph_undirected_2021-06-01.graphml")
    leiden,louvain=load_com(path_com,names_com,dtype_com)
    if MAKE:
        positions=position_creation(G,path_to_read= DATA_DIR + 'position.json')
    else:
        positions=pos_reading(DATA_DIR + 'position.json')
    drawing(drawing_params(positions,G,com2col,leiden),path_to_draw=DATA_DIR+ "network_map.pdf")


if __name__ == "__main__":
    main()

