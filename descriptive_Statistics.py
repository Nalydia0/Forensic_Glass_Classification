import pandas as pd

# Load in our CSV file and Produce some Descriptive Statistics.
##Try to make this a little formatted
glass_dataframe = pd.read_csv("glass.csv", names=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Target'])
describe_table = glass_dataframe.describe().reset_index().rename(columns={'index': ''})

# Print the table
print(describe_table.to_string(index=False))