import re
import pandas as pd

# clean Thai character
df = pd.read_excel("test_data_0904.xlsx")
df["ORIGINAL MODEL NAME"] = [re.sub(r'[^\x00-\x7F]+', '', text) for text in df["ORIGINAL MODEL NAME"]]

# remove space at the end
df = pd.read_excel("Sample Data_7.xlsx")
df['KATABAN'] = df['KATABAN'].str.rstrip()

# Export DataFrame to Excel
output_file = 'Sample Data_10.xlsx'
df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")

