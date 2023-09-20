import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the saved model
model_output_dir = 'C:/Users/Lenovo/Desktop/Web Scraping/bart_fixlrre_saved_model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = BartForConditionalGeneration.from_pretrained(model_output_dir).to(device)

# Initialize BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Load the new test dataset
new_test_df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/promoter_data_with_KATABAN_1.xlsx")

# Define the number of chunks
num_chunks = 10
chunk_size = len(new_test_df) // num_chunks

# Initialize an empty DataFrame to store the results
predicted_df = pd.DataFrame()

# Divide the new_test_df into smaller chunks and predict SKU names
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(new_test_df)
    chunk_df = new_test_df.iloc[start_idx:end_idx].copy()

    # Tokenize the input model names from the current chunk
    chunk_model_names = chunk_df["Competitor Model Name"].tolist()
    chunk_input_ids = tokenizer(chunk_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
    chunk_input_ids = chunk_input_ids.input_ids.to(device)

    # Generate predictions for the current chunk
    loaded_model.eval()
    chunk_output = loaded_model.generate(chunk_input_ids, max_length=16)
    chunk_output_text = tokenizer.batch_decode(chunk_output, skip_special_tokens=True)

    # Add Predicted SKU Name to the current chunk
    chunk_df['Predicted SKU Name'] = chunk_output_text

    # Append the current chunk's results to the predicted_df
    predicted_df = pd.concat([predicted_df, chunk_df], ignore_index=True)

# Add column indicating whether the prediction is correct
predicted_df['correct'] = predicted_df['Predicted SKU Name'] == predicted_df['KATABAN']
predicted_df['correct'] = predicted_df['correct'].astype(int)

# Calculate and print accuracy
correct_predictions = (predicted_df['Predicted SKU Name'] == predicted_df['KATABAN']).sum()
total_predictions = len(predicted_df)
accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Export the new_test dataset with Predicted SKU Name to Excel
new_test_output_file = 'C:/Users/Lenovo/Desktop/Web Scraping/Competitor_Model_List_Predicted_All.xlsx'
predicted_df.to_excel(new_test_output_file, index=False)
print(f"New test data with predictions exported to {new_test_output_file}")
