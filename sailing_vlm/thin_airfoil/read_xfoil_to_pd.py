import pandas as pd

df =pd.read_csv("example.csv",  skiprows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],  sep='\s+')
# df =pd.read_csv("example2.csv",  skiprows=[1],  sep='\s+')
# df =pd.read_csv("example3.csv", sep='\s+')

print("bye")

