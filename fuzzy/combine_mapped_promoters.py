import pandas as pd

# Sample DataFrames
# Replace these with your actual DataFrames
df1 = pd.read_excel("Competitor Model List.xlsx")
df2 = pd.read_excel("promoter_data_mapped.xlsx")

# Merge the DataFrames based on 'Competitor Model Name'
merged_df = pd.merge(df1, df2, on='Competitor Model Name', how='left')

# Export DataFrame to Excel
output_file = 'Competitors model mapping.xlsx'
merged_df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")

