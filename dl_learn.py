import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

import sys

iris_datasets=datasets.load_iris()
iris_pd=pd.DataFrame(iris_datasets.data,columns=iris_datasets.feature_names)
iris_pd["target"]=iris_datasets.target

if len(sys.argv)>1 :
    if sys.argv[1]=="info":
        print(iris_pd.info())
    elif sys.argv[1]=="describe":
        print(iris_pd.describe())
    elif sys.argv[1]=="query":
        if len(sys.argv)<3:
            print("please input query")
        else:
            print(iris_pd.query(sys.argv[2]))
    elif sys.argv[1]=="-h":
        print("""options:
    info:info
    describe:describe
    query:query String
""")