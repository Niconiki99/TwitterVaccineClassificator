"""Build graphs from retweets.

The overall goal is the creation of different types of graphs representing retweet interactions in a social media dataset.
Data Loading:
The load_data function loads a CSV file containing social media data into a Pandas DataFrame. It filters the rows based on a provided deadline timestamp, which acts as a cutoff for the dataset.
Graph Construction:
The compute_graph function extracts tweet-retweet pairs from the DataFrame, organizing the data into a new DataFrame with columns for the source, hyperlink, and target of retweets.
Hypergraph Construction:
The write_hypergraph function builds a hypergraph from the retweet data. It uses sparse matrices to represent connections between retweet sources/targets and hyperlinks. The hypergraph is constructed by mapping unique users to numerical indices.
Graph Loading:
The load_graph function loads the hypergraph components (head and tail matrices, along with user information) from a specified directory.
Largest Connected Component Extraction:
The extract_largest_component function extracts the largest connected component from the hypergraph, removing users and corresponding retweets from smaller components.

NetworkX and External Dependencies:
The code uses the NetworkX library for graph-related operations.
External dependencies include numpy, pandas, scipy, and pathlib.
"""
from __future__ import annotations

import pathlib
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from configuration_params import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR,NETWORK_DATA,NETPATH
NETPATH.mkdir(parents=True, exist_ok=True)


def load_data(deadline: str,path: str) -> pd.DataFrame:
    """
    Load the full dataset and filter rows based on a given deadline.

    Parameters:
    - deadline (str): The deadline timestamp to filter the dataset.
    - path (str): The file path to the CSV dataset.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the dataset.

    Notes:
    - The function reads a CSV file into a Pandas DataFrame, sets the 'id' column as the index,
      and performs data type conversion for specific columns.
    - Missing values in certain columns are specified as `["", "[]"]`.
    - The 'created_at' column is parsed as datetime.
    - Rows with a 'created_at' timestamp greater than or equal to the provided 'deadline'
      are filtered out.
    """
    df_full = pd.read_csv(
        path,
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
    """
    Extract (tweet, retweets) pairs from a DataFrame and organize the data.

    Parameters:
    - df_full (pd.DataFrame): The input DataFrame containing the full dataset.

    Returns:
    - pd.DataFrame: A DataFrame containing (source, hyperlink, target) pairs representing retweets.

    Notes:
    - The function filters rows from the input DataFrame where 'retweeted_status.id' is not NaN.
    - It selects relevant columns ('user.id', 'retweeted_status.id', 'retweeted_status.user.id') for retweets.
    - The columns are then renamed to represent source, hyperlink, and target users.
    - Hyperlinks are assigned numerical indices starting from 0.
    - The function prints the number of retweets before returning the resulting DataFrame.
    """
    retweets = df_full.dropna(subset=["retweeted_status.id"])[
        ["user.id", "retweeted_status.id", "retweeted_status.user.id"]
    ]

    # use meaningful headers
    retweets.columns = ["source", "hyperlink", "target"]

    # I want hyperlinks counted from 0, 1... as this will become the column index.
    hyperlinks = {k: i for i, k in enumerate(retweets["hyperlink"].unique())}
    retweets["hyperlink"] = retweets["hyperlink"].map(lambda x: hyperlinks[x])
    print("Num of retweets", len(retweets))
    return retweets


def write_hypergraph(retweets: pd.DataFrame, deadline: pd.Timestamp,savenames:list=False,write:Bool=True) -> tuple:
    """
    Write down the hypergraph.

    Parameters:
    - retweets (pd.DataFrame): DataFrame containing information about retweets, with columns "source", "target", and "hyperlink".
    - deadline (pd.Timestamp): Timestamp representing the deadline for building the hypergraph.
    - savenames (list, optional): List of filenames for saving the hypergraph components. If not provided, default names are used.
    - write (bool, optional):Define if saving the hypergraph produced or not.

    Returns:
    - the product between the tail and the head matrix and the list of users.

    This function builds a hypergraph from retweet data and saves its components. The hypergraph is represented by two sparse matrices:
    - The "head" matrix captures connections from retweet targets to hyperlinks.
    - The "tail" matrix captures connections from retweet sources to hyperlinks.

    The hypergraph is constructed by mapping unique users to numerical indices and using sparse matrices to represent connections between users and hyperlinks.

    The function also extracts the largest connected component from the hypergraph, saving the resulting matrices and user information.

    """
    if(write):
        print("Building hyprgraph for", deadline.date())
        if(not(savenames)):
            savenames=[f"hyprgraph_{deadline.date()}_head.npz",f"hyprgraph_{deadline.date()}_tail.npz",f"hyprgraph_{deadline.date()}_usermap.csv.gz"]

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
    
    if write:
        sparse.save_npz(NETPATH / savenames[0], head)
        sparse.save_npz(NETPATH / savenames[1], tail)
        users.to_csv(NETPATH / savenames[2])

    return tail @ head.T, users


def load_graph(deadline: pd.Timestamp,path : pathlib.PosixPath,savenames:list=False) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.Series]:
    """
    Load head, tail, and usermap matrices representing a hypergraph.

    Parameters:
    - deadline (pd.Timestamp): Timestamp representing the deadline for which to load the hypergraph.
    - path (pathlib.PosixPath): Path to the directory containing the hypergraph components.
    - savenames (list, optional): List of filenames for loading the hypergraph components. If not provided, default names are used.

    Returns:
    - tuple (sparse.csr_matrix, sparse.csr_matrix, pd.Series): A tuple containing the head matrix, tail matrix, and a Series representing user indices.

    This function loads the head and tail matrices along with user information from the specified directory. 
    The hypergraph components are assumed to be saved in compressed sparse row (CSR) format for efficiency.

    Examples:
    ```python
    deadline = pd.Timestamp(datetime(2024, 1, 6))
    path_to_hypergraph = pathlib.PosixPath("/path/to/hypergraph_data/")
    head_matrix, tail_matrix, users_series = load_graph(deadline, path_to_hypergraph)
    ```
    """
    if(not(savenames)):
        try:
            savenames=[f"hyprgraph_{deadline.date()}_head.npz",f"hyprgraph_{deadline.date()}_tail.npz",f"hyprgraph_{deadline.date()}_usermap.csv.gz"]
        except AttributeError:
            savenames=[f"hyprgraph_{deadline}_head.npz",f"hyprgraph_{deadline}_tail.npz",f"hyprgraph_{deadline}_usermap.csv.gz"]
    try:
        head = sparse.load_npz(path / savenames[0])
        tail = sparse.load_npz(path / savenames[1])
        users = pd.read_csv(
        path / savenames[2],
        index_col=0,
        dtype="int64",
    )["0"]
    except TypeError:
        head = sparse.load_npz(path + "/"+savenames[0])
        tail = sparse.load_npz(path + "/"+savenames[1])
        users = pd.read_csv(
        path +"/"+savenames[2],
        index_col=0,
        dtype="int64",
    )["0"]

    

    return tail, head, users


def extract_largest_component(tail: sparse.csr_matrix, head: sparse.csc_matrix) -> (sparse.csr_matrix, sparse.csr_matrix):
    """
    This function extracts the largest connected component from a hypergraph represented by its tail and head matrices. 
    It removes users from smaller components and corresponding retweets involving those smaller components.

    The function uses the connected components algorithm to identify the largest component, projects the hypergraph matrices onto the users in the largest component, and removes retweets involving users from smaller components.

    Parameters:
    - tail (sparse.csr_matrix): Sparse matrix representing the tail connections in the hypergraph.
    - head (sparse.csc_matrix): Sparse matrix representing the head connections in the hypergraph.

    Returns:
    - tuple (sparse.csr_matrix, sparse.csr_matrix, np.ndarray): A tuple containing the tail matrix, head matrix, and an array representing user indices in the largest connected component.
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
    """
    Toggle format from string to pandas Timestamp.

    Parameters:
    - date (str or pd.Timestamp): A date represented as a string or pandas Timestamp.

    Returns:
    - pd.Timestamp or str: If the input is a string, returns a pandas Timestamp; if the input is already a pandas Timestamp, returns its string representation.

    This function allows for flexible handling of date formats. If the input 'date' is a string, it converts it to a pandas Timestamp with a specific time (00:00:00) and timezone offset (+02). If the input is already a pandas Timestamp, it returns its string representation in the format 'YYYY-MM-DD'.

    """
    if isinstance(date, str):
        return pd.Timestamp(date + "T00:00:00+02")
    return date.isoformat().split()[0].split("T")[0]


def main(deadline: pd.Timestamp) -> None:
    """Do the main."""
    print("============")
    print(parse_date(deadline))
    print("============")
    path=LARGE_DATA_DIR+"df_full.csv.gz"
    retweets = compute_graph(load_data(deadline,path))
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
    from configuration_params import deadline
    main(parse_date(deadline))