import pandas as pd
import pytest
import numpy as np
from scipy import sparse
from build_graphs import load_data,compute_graph,write_hypergraph
from DIRS import TEST_PATH

def generating_path():
    path=TEST_PATH+"/df_full.csv.gz"
    deadline=pd.Timestamp("2021-06-01" + "T00:00:00+02")
    return path,deadline

def generating_simple_case():
    #BUILDING AN EXAMPLE DATASET TO TEST THE CONNECTED COMPONENT ANALYSIS
    id=["0","1","2","3"]
    hyperlink=["0","1","2","3"]
    source=["3","0","1","2"]
    target=["1","2","3","0"]
    dict={"id":id,"source":source,"hyperlink":hyperlink,"target":target}
    retweet_simple=pd.DataFrame(dict)
    retweet_simple.set_index("id")
    return retweet_simple

########## TEST FUNCTIONS ##########
def test_load_data():
    path,deadline=generating_path()
    df=load_data(deadline,path)
    assert isinstance(df, pd.DataFrame)


def test_compute_graph():
    path,deadline=generating_path()
    df=load_data(deadline,path)
    assert len(compute_graph(df))==len(df["retweeted_status.id"].dropna())


def test_write_hypergraph():
    path,deadline=generating_path()
    retweet=compute_graph(load_data(deadline,path))
    matrix,users=write_hypergraph(retweet,deadline,write=False)
    assert isinstance(users,pd.Series)
    assert isinstance(matrix,sparse.csr_matrix)

def test_write_hypergraph_simple_case():
    retweet_simple=generating_simple_case()
    deadline=''
    matrix,users=write_hypergraph(retweet_simple,deadline,write=False)
    assert matrix.shape==(2,2)
