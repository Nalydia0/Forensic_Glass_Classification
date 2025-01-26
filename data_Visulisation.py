import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
#Load in the glass dataframe 

columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Target']
glass_dataframe = pd.read_csv("glass.csv",names = columns)
# Count the occurrences of each glass type

glass_counts = glass_dataframe['Target'].value_counts().sort_index()

# Plotting this in a bar chart
plt.figure(figsize=(10, 6))
glass_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Types of Glass')
plt.xlabel('Glass Type')
plt.ylabel('Count')
plt.show()

# Create a DataFrame from the counts of the glass types
df = pd.DataFrame({'Number': glass_counts.values}, index=glass_counts.index)

# Sort the DataFrame by the "Number" column in descending order
df_sorted = df.sort_values(by='Number', ascending=False)

# Rename the index and "Target" column to "Type of Glass" and "Number" respectively
df_sorted.index.name = 'Type of Glass'
df_sorted.rename(columns={'Target': 'Number'}, inplace=True)

# Display the sorted counts as a formatted tabl
print(tabulate(df_sorted.reset_index(), headers='keys', tablefmt='fancy_grid', showindex=True))