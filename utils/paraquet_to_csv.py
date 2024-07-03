import pandas as pd
import duckdb
test = pd.read_parquet('0000.parquet', engine='pyarrow')
# test = pd.read_csv('train1.csv')
lens = len(test)
_ = 0
# while _ != lens:
#     test.loc[_,'target'] = "\s "+test.loc[_]['target']+" \e"
#     _ +=1
test.to_csv("train1.csv",index=False)
