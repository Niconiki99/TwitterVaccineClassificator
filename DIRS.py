import pathlib
p=pathlib.Path()
this=p.cwd()
TRANSFORMERS_CACHE_DIR=str(this/"Trasf_cache")
DATA_DIR=str(this/"Data/Data/")
LARGE_DATA_DIR=str(this/"Data/Large_Data/")
NETWORK_DATA=str(this/"Data/Network_Data")
TEST_PATH=str(this/"Data/Test")
