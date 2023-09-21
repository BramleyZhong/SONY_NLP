import pandas as pd

df = pd.read_excel("predict.xlsx")

def calculate_accuracy(df, column1, column2):
    total_rows = len(df)
    correct_matches = sum(df[column1] == df[column2])
    accuracy = correct_matches / total_rows
    return accuracy

# Example usage:
# Assuming you have a DataFrame named 'df' and two columns 'Column1' and 'Column2'
# You can call the function like this:
accuracy = calculate_accuracy(df, column1='KATABAN', column2='predicted model')
print(f"Accuracy: {accuracy:.2%}")