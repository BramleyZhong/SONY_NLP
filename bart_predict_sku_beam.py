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
new_test_df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/Sample Data_11.xlsx")

# Tokenize the input model names from the new test dataset
new_test_model_names = new_test_df["ORIGINAL MODEL NAME"].tolist()
new_test_input_ids = tokenizer(new_test_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
new_test_input_ids = new_test_input_ids.input_ids.to(device)

# Define the number of beams for beam search
num_beams = 4  # You can adjust this number based on your preferences

# Generate predictions for the new test dataset using beam search
loaded_model.eval()
new_test_output = loaded_model.generate(
    new_test_input_ids,
    max_length=16,
    num_beams=num_beams,  # Adding beam search
    early_stopping=True,  # Stop generation when all beams have reached the end
    num_return_sequences=1  # Return only the top sequence
)
new_test_output_text = tokenizer.batch_decode(new_test_output, skip_special_tokens=True)

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
new_test_output_file = 'C:/Users/Lenovo/Desktop/Web Scraping/nnew_test_data_with_predictions_wbeam_1.xlsx'
new_test_df.to_excel(new_test_output_file, index=False)
print(f"New test data with predictions (beam search) exported to {new_test_output_file}")

