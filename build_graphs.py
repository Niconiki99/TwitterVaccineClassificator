"""Build graphs from retweets.

It will build:
    1. retweet network (symmetric)
    2. retweet network (directed)
    3. return hyprgraph (directed, head + tail)
"""
from __future__ import annotations

import pathlib
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR,NETWORK_DATA

NETPATH = pathlib.Path(NETWORK_DATA)
NETPATH.mkdir(parents=True, exist_ok=True)


def load_data(deadline: str) -> pd.DataFrame:
    """Load the full dataset."""
    df_full = pd.read_csv(
        LARGE_DATA_DIR + "df_full.csv.gz",
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
        # nrows=100000,  # uncomment to test the code on a short dataset
    )
    return df_full[df_full.created_at < deadline]


def compute_graph(df_full: pd.DataFrame) -> pd.DataFrame:
    """Load the whole dataset and compute the (tweet, retweets) pairs."""
    # columns:
    # id,created_at,text,user.id,user.screen_name,place,url,
    #      retweeted_status.id,retweeted_status.user.id,retweeted_status.url,
    #      annotation,user_annotation,lang

    # filter only tweets before a deadline.

    # users that retweet
    retweets = df_full.dropna(subset="retweeted_status.id")[
        ["user.id", "retweeted_status.id", "retweeted_status.user.id"]
    ]

    # use meaningful headers
    retweets.columns = ["source", "hyperlink", "target"]

    # I want hyperlinks counted from 0, 1... as this will become the column index.
    hyperlinks = {k: i for i, k in enumerate(retweets["hyperlink"].unique())}
    retweets["hyperlink"] = retweets["hyperlink"].map(lambda x: hyperlinks[x])
    print("Num of retweets", len(retweets))
    return retweets


def write_hypergraph(retweets: pd.DataFrame, deadline: pd.Timestamp) -> None:
    """Write down the hyprgraph."""
    print("Building hyprgraph for", deadline.date())

    users = set(retweets["source"]) | set(retweets["target"])
    users = {u: i for i, u in enumerate(users)}

    hg_links = retweets["hyperlink"]
    hg_source = retweets["source"].map(lambda x: users[x])
    hg_target = retweets["target"].map(lambda x: users[x])

    tail = sparse.coo_matrix(
        (np.ones(len(retweets)), (hg_source, hg_links)),
        shape=(len(users), len(retweets)),
        dtype=int,
    ).tocsr()
    print("Tail", tail.shape)
    head = sparse.coo_matrix(
        (np.ones(len(retweets)), (hg_target, hg_links)),
        shape=(len(users), len(retweets)),
        dtype=int,
    ).tocsr()
    print("Head", head.shape)

    # only get the largest component
    tail, head, comp_indx = extract_largest_component(tail, head)

    users = pd.Series(list(users.keys()), index=list(users.values()))
    users = users[comp_indx].reset_index(drop=True)
    print(users.shape, tail.shape, head.shape)

    sparse.save_npz(NETPATH / f"hyprgraph_{deadline.date()}_head.npz", head)
    sparse.save_npz(NETPATH / f"hyprgraph_{deadline.date()}_tail.npz", tail)

    users.to_csv(NETPATH / f"hyprgraph_{deadline.date()}_usermap.csv.gz")

    return tail @ head.T, users


def load_graph(
    deadline: pd.Timestamp
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.Series]:
    """Load head tail and usermap."""
    head = sparse.load_npz(NETPATH / f"hyprgraph_{deadline}_head.npz")
    tail = sparse.load_npz(NETPATH / f"hyprgraph_{deadline}_tail.npz")

    users = pd.read_csv(
        NETPATH / f"hyprgraph_{deadline}_usermap.csv.gz",
        index_col=0,
        dtype="int64",
    )["0"]

    return tail, head, users


def extract_largest_component(
    tail: sparse.csr_matrix, head: sparse.csc_matrix
) -> (sparse.csr_matrix, sparse.csr_matrix):
    """Extract the largest component.

    remove users from smaller componets and retweets that involve those smaller components.
    """
    rtw_net = tail @ head.T
    print("Full adj", rtw_net.shape)

    n_comps, components = sparse.csgraph.connected_components(rtw_net, directed=False)

    largest_components = Counter(components).most_common(1)
    largest_component, new_nn = largest_components[0]
    largest_component = np.argwhere(components == largest_component).flatten()
    print(
        f"Largest component with {new_nn} users ({100 * new_nn/tail.shape[0]:5.2f} %)."
    )

    # projector to users in the largest component
    largest_component_proj = sparse.coo_matrix(
        (
            np.ones_like(largest_component),
            (np.arange(len(largest_component)), largest_component),
        ),
        shape=(new_nn, tail.shape[0]),
    ).tocsr()
    tail = largest_component_proj @ tail
    head = largest_component_proj @ head

    _, retweets_to_keep = np.nonzero(np.asarray(tail.sum(0)) * np.asarray(head.sum(0)))
    print(f"Retweets in smaller componets: {tail.shape[1] - retweets_to_keep.shape[0]}")
    retweets_to_keep_proj = sparse.coo_matrix(
        (
            np.ones_like(retweets_to_keep),
            (retweets_to_keep, np.arange(len(retweets_to_keep))),
        ),
        shape=(tail.shape[1], len(retweets_to_keep)),
    )
    tail = tail @ retweets_to_keep_proj
    head = head @ retweets_to_keep_proj
    print("Tail", tail.shape)
    print("Head", head.shape)

    return tail, head, largest_component


def parse_date(date: str | pd.Timestamp) -> pd.Timestamp | str:
    """Toggle format from str to pd.Timestamp."""
    if isinstance(date, str):
        return pd.Timestamp(date + "T00:00:00+02")
    return date.isoformat().split()[0].split("T")[0]


def main(deadline: pd.Timestamp) -> None:
    """Do the main."""
    print("============")
    print(parse_date(deadline))
    print("============")

    retweets = compute_graph(load_data(deadline))
    adj, users = write_hypergraph(retweets, deadline)

    # Directed graph
    graph = nx.from_scipy_sparse_array(
        adj, create_using=nx.DiGraph, edge_attribute="weight"
    )
    # node label is saved in hyprgraph_deadline_usermap.csv.gz
    nx.relabel_nodes(graph, mapping=dict(zip(users.index, users)),copy=False)
    nx.write_graphml_lxml(
        graph,
        NETPATH / f"retweet_graph_directed_{parse_date(deadline)}.graphml",
    )
    nx.write_graphml(
        graph.to_undirected(),
        NETPATH / f"retweet_graph_undirected_{parse_date(deadline)}.graphml",named_key_ids=True
    )


if __name__ == "__main__":
        main(parse_date("2021-06-01"))