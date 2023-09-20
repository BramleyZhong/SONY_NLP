import pandas as pd
import time
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the saved model
model_output_dir = 'C:/Users/Lenovo/Desktop/Web Scraping/bart_combined_saved_model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = BartForConditionalGeneration.from_pretrained(model_output_dir).to(device)

# Initialize BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Load the new test dataset
new_test_df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/Combined_test_data.xlsx")

# Tokenize the input model names and BRAND from the new test dataset
new_test_model_names = new_test_df["ORIGINAL MODEL NAME"].tolist()
new_test_brands = new_test_df["BRAND"].tolist()

# Tokenize BRAND
brand_input_ids = tokenizer(new_test_brands, return_tensors='pt', padding=True, truncation=True, max_length=16)
brand_input_ids = brand_input_ids.input_ids.to(device)

# Tokenize the input model names
new_test_input_ids = tokenizer(new_test_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
new_test_input_ids = new_test_input_ids.input_ids.to(device)

# Record the start time
start_time = time.time()

# Generate predictions for the new test dataset
loaded_model.eval()

# Create combined input by concatenating model names and brands
combined_input_ids = torch.cat((new_test_input_ids, brand_input_ids), dim=1)

new_test_output = loaded_model.generate(combined_input_ids, max_length=16)
new_test_output_text = tokenizer.batch_decode(new_test_output, skip_special_tokens=True)

# Record the end time
end_time = time.time()

# Calculate and print the time taken for prediction
prediction_time = end_time - start_time
print(f"Time taken for prediction: {prediction_time:.2f} seconds")

# Add Predicted SKU Name to the new test dataset
new_test_df['Predicted SKU Name'] = new_test_output_text

# Add column indicating whether the prediction is correct
new_test_df['correct'] = new_test_df['Predicted SKU Name'] == new_test_df['KATABAN']
new_test_df['correct'] = new_test_df['correct'].astype(int)

# Calculate and print accuracy
correct_predictions = (new_test_df['Predicted SKU Name'] == new_test_df['KATABAN']).sum()
total_predictions = len(new_test_df)
accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Export the new test dataset with Predicted SKU Name to Excel
new_test_output_file = 'C:/Users/Lenovo/Desktop/Web Scraping/matched_combined_new_data_6.xlsx'
new_test_df.to_excel(new_test_output_file, index=False)
print(f"New test data with predictions exported to {new_test_output_file}")

# Free up GPU memory
torch.cuda.empty_cache()

