"""Starting from the graphs, compute the community structures.
This code snippet contains functions for computing partitions, transition matrices, and simplifying community structures in a hypergraph. 
It utilizes various libraries such as igraph, networkx, and sknetwork for community detection and analysis. 
Defining Community Partitioning for Hypergraph Core:
partition_core function is defined to compute the community partition of the core of a hypergraph; removing dangling nodes
Defining Community Partitioning for Graph:
partition function is defined to compute community partitions for a given graph using various methods (e.g., Louvain, Leiden).
Simplifying Community Structure:
simplify_community_struct function reduces the number of communities based on either a community size cutoff or a coverage ratio.
"""

from __future__ import annotations

from time import time

import pathlib
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import sknetwork
from scipy import sparse
from configobj import ConfigObj
from build_graphs import  load_graph , parse_date


def partition_core(
    tail: sparse.spmatrix, head: sparse.spmatrix, usermap: pd.Series, kind="sk_louvain"
) -> pd.Series:
    """
    Compute the partition of the core of the graph.
    The function computes the core of the hypergraph by removing dangling nodes.
    It then computes the community structure using the specified algorithm (default is Louvain).
    Dangling nodes are added to the community of their neighboring core nodes.

    Parameters:
    - tail (sparse.spmatrix): Sparse matrix representing the tail of the hypergraph.
    - head (sparse.spmatrix): Sparse matrix representing the head of the hypergraph.
    - usermap (pd.Series): Series mapping user indices to user IDs.
    - kind (str): Algorithm for community detection (default is "sk_louvain").

    Returns:
    - pd.Series: Series containing the community assignments for each user in the core.
    """
    adjacency = (tail @ head.T).tocsr()

    # take the binary version
    # this is needed to test weather a user tweeted to or retweeted from
    # only one other user
    adjacency.data = np.ones(adjacency.nnz)
    retweeters = adjacency.sum(0).A
    tweeters = adjacency.sum(1).A.T

    # users that either have no tweets and retweets just one other user or
    # have tweets that get retweeted only from another user and no retweets.
    # In term of topology, these are dangling nodes with either one outgoing or ingoing link
    dangling = np.logical_or(
        np.logical_and(retweeters == 0, tweeters == 1),
        np.logical_and(retweeters == 1, tweeters == 0),
    )
    core = np.argwhere(~dangling)[:, 1]
    dangling = np.argwhere(dangling)[:, 1]

    proj_dangling = sparse.coo_matrix(
        (np.ones_like(dangling), (dangling, np.arange(len(dangling)))),
        shape=(tail.shape[0], len(dangling)),
    ).tocsr()
    dangling_links = proj_dangling.T @ adjacency + (adjacency @ proj_dangling).T
    i, j, _ = sparse.find(dangling_links)
    # map from dangling indexes to their neighbors indexes.
    dangling_map = {dangling[_i]: _j for _i, _j in zip(i, j)}

    proj_core = sparse.coo_matrix(
        (np.ones_like(core), (core, np.arange(len(core)))),
        shape=(tail.shape[0], len(core)),
    ).tocsr()
    core_adj = proj_core.T @ (tail @ head.T).tocsr() @ proj_core

    # compute the partition structure on the core of the network
    core_partition = partition(core_adj, kind=kind)

    # map the index to the original index in adjacency
    core_partition.index = core_partition.index.map(pd.Series(core))
    # append dangling nodes with the original indexes (keys) and the community
    # of the neighbor in the core (core_partition[c])
    core_partition = pd.concat(
        [
            core_partition,
            pd.Series(
                [core_partition[c] for c in dangling_map.values()],
                index=dangling_map.keys(),
            ),
        ]
    )

    core_partition.index = core_partition.index.map(usermap)

    return core_partition


def partition(adj: sparse.spmatrix, kind: str = "louvain", usermap: pd.Series | None = None) -> pd.Series:
    """
    Compute partitions with various methods. In particular using Louvain and Leiden methods:
    If 'usermap' is provided, it maps the indices to user IDs in the output Series.
    
    Parameters:
    - adj (sparse.spmatrix): Sparse adjacency matrix representing the graph.
    - kind (str): Method for community detection (default is "louvain").
    - usermap (pd.Series | None): Series mapping user indices to user IDs (optional).
    
    Returns:
    - pd.Series: Series containing the community assignments for each node.
    """
    print("Computing",kind)
    t0 = time()
    if kind == "louvain":
        p = nx.community.greedy_modularity_communities(
            # use the undirected form.
            nx.from_scipy_sparse_array(
                adj, create_using=nx.Graph, edge_attribute="weight"
            ),
            weight="weight",
        )
    elif kind == "sk_louvain":
        # Much faster than networkx
        louvain = sknetwork.clustering.Louvain()
        p = louvain.fit_predict(adj)
        p = pd.Series(p, name="sk_louvain")
    elif kind == "leiden":
        # optimize modularity with leiden in igraph
        graph_ig = sparse2igraph(adj, directed=False)
        p = graph_ig.community_leiden(objective_function="modularity", weights="weight")

    if kind ==  "leiden":
        # convert igraph result to pd.Series
        p = {u: ip for ip, _p in enumerate(p) for u in _p}
        p = pd.Series(p.values(), index=p.keys())
    print("Elapsed time", time() - t0)
    print(p.value_counts())

    if usermap is not None:
        p.index = p.index.map(usermap)

    return p





def plot_comm_size(parts: pd.DataFrame, path:pathlib.Path| str) -> None:
    """
    Plot the community sizes.

    Parameters:
    - parts (pd.DataFrame): DataFrame containing community assignments for nodes.
    - path (pathlib.PosixPath|str): Where to save the plots.
    Returns:
    - None
    """
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(
        ncols=len(parts.columns), sharey=True, figsize=(3 * len(parts.columns), 5)
    )

    for ax, part in zip(axs, parts.columns):
        print(part)
        counts = parts[part].value_counts()
        print(counts)
        counts = counts.sort_values(ascending=False).cumsum()
        print(counts)
        counts /= len(parts[part])
        ax.scatter(range(len(counts)), counts)
        ax.semilogx()
        ax.set_title(part)
        ax.set_xlim(1, 20)
        ax.axhline(0.9)
    axs[0].set_ylabel("Cumulative ratio.")
    plt.savefig(path/"plot_community_sizes.pdf")
    plt.close()


def simplify_community_struct(
    community: pd.Series, comm_size: int = 0, coverage: float = 0.0
) -> pd.Series:
    """
    Reduce the number of communities based on either a community size cutoff or a coverage ratio.
    The choice depends on of if 'comm_size' is specified or if 'coverage' is specified

    Parameters:
    - community (pd.Series): Series containing community assignments for nodes.
    - comm_size (int): Community size cutoff. Communities smaller than this size will be removed (default is 0).
    - coverage (float): Coverage ratio. Larger communities covering at least this ratio of the network will be kept (default is 0.0).

    Returns:
    - pd.Series: Simplified community assignments after reduction.
    """
    counts = community.value_counts().sort_values(ascending=False)
    if comm_size > 0:
        # remove communities smaller than comm_size
        keep = counts[counts > comm_size].index
    elif coverage > 0.0:
        # keep larger communities that cover at least coverage ratio of the network
        keep = pd.Index(counts.cumsum() / len(community))
        keep_num = len(keep[keep <= coverage]) + 1
        keep = pd.Series(keep).iloc[:keep_num].index

    new_parts = {v: i for i, v in enumerate(keep)}
    return community.map(lambda x: new_parts.get(x, len(new_parts)))


def sparse2igraph(adjacency: sparse.spmatrix, **kwargs: dict) -> igraph.Graph:
    """
    Convert a sparse matrix to an igraph Graph.

    Parameters:
    - adjacency (sparse.spmatrix): Sparse adjacency matrix representing the graph.
    - **kwargs (dict): Additional keyword arguments to pass to the igraph.Graph constructor.

    Returns:
    - igraph.Graph: An igraph Graph representation of the input sparse matrix.
    """
    i, j, v = sparse.find(adjacency)
    graph = igraph.Graph(edges=zip(i, j), **kwargs)
    graph.es["weight"] = v
    return graph


def main() -> None:
    """Do the main."""
    config= ConfigObj("config.txt")
    deadline=parse_date(config["DEADLINE"]["deadline"])
    path=config["READING_PARAMS"]["DF_FULL"]["path"]
    TRANSFORMERS_CACHE_DIR=config["DIRS"]["TRANSFORMERS_CACHE_DIR"]
    LARGE_DATA_DIR=config["DIRS"]["LARGE_DATA_DIR"]
    NETWORK_DATA=config["DIRS"]["NETWORK_DATA"]
    DATA_DIR=config["DIRS"]["DATA_DIR"]
    NETPATH = pathlib.Path(NETWORK_DATA)
    print("============")
    print(parse_date(deadline))
    print("============")
    DATAPATH = pathlib.Path(DATA_DIR)
    DATAPATH.mkdir(parents=True, exist_ok=True)
    path=NETPATH
    tail, head, usermap = load_graph(parse_date(deadline),path)
    adj = tail @ head.T
    p = pd.DataFrame()

    pp = partition_core(tail, head, usermap, kind="leiden")
    p["leiden"] = pp
    p.index = pp.index

    p["louvain"] = partition_core(tail, head, usermap, kind="sk_louvain")

    plot_comm_size(p,DATAPATH)

    for part in p.columns:
        p[part + "_5000"] = simplify_community_struct(p[part], comm_size=5000)
        p[part + "_90"] = simplify_community_struct(p[part], coverage=0.9)

    p.to_csv(DATAPATH / f"communities_{deadline}.csv.gz")


if __name__ == "__main__":
    main()
