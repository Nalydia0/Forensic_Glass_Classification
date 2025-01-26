import pandas as pd
##load in our dataset 
## Setup some column names
columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Target']
glass_dataframe = pd.read_csv("glass.csv",names = columns)
glass_dataframe