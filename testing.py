import pandas as pd
import pytest
import numpy as np
from scipy import sparse
from build_graphs import load_data,compute_graph,write_hypergraph,load_graph
from build_communities import partition_core,simplify_community_struct
from DIRS import TEST_PATH

def generating_path():
    path=TEST_PATH
    deadline=pd.Timestamp("2021-06-01" + "T00:00:00+02")
    return path,deadline,

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
    
def generate_simple_adj():
    x=[1,0,3,2,3,0]
    y_=[0,1,2,3,0,3]
    y=[0,1,2,3]
    adj=sparse.coo_matrix((np.ones(6),(x,y_)),shape=(4,4))
    id=sparse.coo_matrix((np.ones(4),(y,y)),shape=(4,4))
    return adj,id
########## TEST FUNCTIONS ##########
def test_load_data():
    path,deadline=generating_path()
    path=path+"/df_full.csv.gz"
    df=load_data(deadline,path)
    assert isinstance(df, pd.DataFrame)


def test_compute_graph():
    path,deadline=generating_path()
    path=path+"/df_full.csv.gz"
    df=load_data(deadline,path)
    assert len(compute_graph(df))==len(df["retweeted_status.id"].dropna())


def test_write_hypergraph():
    path,deadline=generating_path()
    path=path+"/df_full.csv.gz"
    retweet=compute_graph(load_data(deadline,path))
    matrix,users=write_hypergraph(retweet,deadline,write=False)
    assert isinstance(users,pd.Series)
    assert isinstance(matrix,sparse.csr_matrix)

def test_write_hypergraph_simple_case():
    retweet_simple=generating_simple_case()
    deadline=''
    matrix,users=write_hypergraph(retweet_simple,deadline,write=False)
    assert matrix.shape==(2,2)

def test_partition_core_type():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    assert isinstance(partition_core(tail,head,usermap),pd.Series)
    assert isinstance(partition_core(tail,head,usermap,kind="leiden"),pd.Series)

def test_partition_core_simple_case():
    adj,id=generate_simple_adj()
    assert len(partition_core(adj,id,pd.Series([0,1,2,3])).unique())==2
    assert len(partition_core(adj,id,pd.Series([0,1,2,3]),kind="leiden").unique())==2

def test_simplify_community_struct_size_lv():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    simplified_com=simplify_community_struct(partition_core(tail,head,usermap),comm_size=50)
    assert (i<51 for i in simplified_com.value_counts().sort_values(ascending=False))

def test_simplify_community_struct_size_ld():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    simplified_com=simplify_community_struct(partition_core(tail,head,usermap,kind="leiden"),comm_size=50)
    assert (i<51 for i in simplified_com.value_counts().sort_values(ascending=False))

def test_simplify_community_coverage_lv_40():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    coms=partition_core(tail,head,usermap)
    simplified_com=simplify_community_struct(coms,coverage=0.4)
    counts=coms.value_counts().sort_values(ascending=False)
    coverage=np.sum(counts.iloc[:len(np.unique(simplified_com.values))-1].values)/len(coms)
    assert (coverage>0.39)

def test_simplify_community_coverage_ld_40():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    coms=partition_core(tail,head,usermap,kind="leiden")
    simplified_com=simplify_community_struct(coms,coverage=0.4)
    counts=coms.value_counts().sort_values(ascending=False)
    coverage=np.sum(counts.iloc[:len(np.unique(simplified_com.values))-1].values)/len(coms)
    assert (coverage>0.39)

def test_simplify_community_coverage_lv_90():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    coms=partition_core(tail,head,usermap)
    simplified_com=simplify_community_struct(coms,coverage=0.9)
    counts=coms.value_counts().sort_values(ascending=False)
    coverage=np.sum(counts.iloc[:len(np.unique(simplified_com.values))-1].values)/len(coms)
    assert (coverage>0.89)

def test_simplify_community_coverage_ld_90():
    path,deadline=generating_path()
    tail, head, usermap = load_graph(deadline,path)
    coms=partition_core(tail,head,usermap,kind="leiden")
    simplified_com=simplify_community_struct(coms,coverage=0.9)
    counts=coms.value_counts().sort_values(ascending=False)
    coverage=np.sum(counts.iloc[:len(np.unique(simplified_com.values))-1].values)/len(coms)
    assert (coverage>0.89)