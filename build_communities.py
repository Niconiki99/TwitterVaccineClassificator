"""Starting from the graphs, compute the community structures."""

from __future__ import annotations

from time import time

import pathlib
import igraph
import networkx as nx
import numpy as np
import pandas as pd
import sknetwork
import tqdm
from scipy import sparse

from build_graphs import NETPATH, load_graph
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR,NETWORK_DATA

DATAPATH = pathlib.Path(DATA_DIR)
DATAPATH.mkdir(parents=True, exist_ok=True)


def partition_core(
    tail: sparse.spmatrix, head: sparse.spmatrix, usermap: pd.Series, kind="sk_louvain"
) -> pd.Series:
    """Compute the partition of the core of the graph.

    Remove dangling nodes and compute the community structure.
    Add the dangling nodes to the neighboring community.
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


def partition(
    adj: sparse.spmatrix, kind: str = "louvain", usermap: pd.Series | None = None
) -> pd.Series:
    """Compute partitions with various methods."""
    print("Computing", kind)

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


def compute_transition_matrix(matrix: sparse.csr_matrix, niter: int = 10000) -> tuple(
    sparse.spmatrix, sparse.spmatrix
):
    r"""Return the transition matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency matrix (square shape)
    niter : int (default=10000)
        number of iteration to converge to the steadystate. (Default value = 10000)

    Returns
    -------
    trans : np.spmatrix
        The transition matrix.
    v0 : np.matrix
        the steadystate
    """
    # marginal
    tot = matrix.sum(0).A1
    # fix zero division
    tot_zero = tot == 0
    tot[tot_zero] = 1
    # transition matrix
    trans = matrix @ sparse.diags(1 / tot)

    # fix transition matrix with zero-sum rows
    trans += sparse.diags(tot_zero.astype(np.float64), offsets=0, shape=trans.shape)

    v0 = matrix.sum(0) + 1
    # v0 = sparse.csr_matrix(np.random.random(matrix.shape[0]))
    v0 = v0.reshape(matrix.shape[0], 1) / v0.sum()
    trange = tqdm.trange(0, niter)
    for i in trange:
        # evolve v0
        v1 = v0.copy()

        v0 = trans.T @ v0
        diff = np.sum(np.abs(v1 - v0))
        if i % 100 == 0:
            trange.set_description(desc=f"diff: {diff}|", refresh=True)
        if diff < 1e-5:
            break
    print(f"TRANS: performed {i + 1} itertions. (diff={diff:2.5f})")

    return trans, v0.A1


def plot_comm_size(parts: pd.DataFrame) -> None:
    """Plot the community sizes."""
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
    plt.savefig("plot_community_sizes.pdf")
    plt.close()


def simplify_community_struct(
    community: pd.Series, comm_size: int = 0, coverage: float = 0.0
) -> pd.Series:
    """Reduce the number of communities.

    The reduction can be done with:
    - a community size cutoff
    - a coverage ratio
    """
    counts = community.value_counts().sort_values(ascending=False)
    if comm_size > 0:
        # remove communities smaller than comm_size
        keep = counts[counts > comm_size].index
    elif coverage > 0.0:
        # keep larger communities that cover at least coverage ratio of the network
        keep = counts.cumsum() / len(community)
        keep_num = len(keep[keep <= coverage]) + 1
        keep = keep.iloc[:keep_num].index

    new_parts = {v: i for i, v in enumerate(keep)}
    return community.map(lambda x: new_parts.get(x, len(new_parts)))


def sparse2igraph(adjacency: sparse.spmatrix, **kwargs: dict) -> igraph.Graph:
    """Convert to igraph."""
    i, j, v = sparse.find(adjacency)
    graph = igraph.Graph(edges=zip(i, j), **kwargs)
    graph.es["weight"] = v
    return graph


def main(deadline: str) -> None:
    """Do the main."""
    path=NETPATH
    tail, head, usermap = load_graph(deadline,path)
    adj = tail @ head.T
    p = pd.DataFrame()

    pp = partition_core(tail, head, usermap, kind="leiden")
    p["leiden"] = pp
    p.index = pp.index

    p["louvain"] = partition_core(tail, head, usermap, kind="sk_louvain")

    plot_comm_size(p)

    for part in p.columns:
        p[part + "_5000"] = simplify_community_struct(p[part], comm_size=5000)
        p[part + "_90"] = simplify_community_struct(p[part], coverage=0.9)

    p.to_csv(DATAPATH / f"communities_{deadline}.csv.gz")


if __name__ == "__main__":
    deadline="2021-06-01"
    main(deadline)
