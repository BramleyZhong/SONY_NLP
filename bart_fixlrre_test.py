import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW

# Load your DataFrame df containing "ORIGINAL MODEL NAME" and "KATABAN" columns
df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/Sample Data_10.xlsx")
model_names = df["ORIGINAL MODEL NAME"].tolist()
sku_names = df["KATABAN"].tolist()

# Create training and validation datasets
train_model_names, val_model_names, train_sku_names, val_sku_names = train_test_split(
    model_names, sku_names, test_size=0.2, random_state=42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

# Tokenize input model names and SKU names
train_input_ids = tokenizer(train_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
val_input_ids = tokenizer(val_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
train_target_ids = tokenizer(train_sku_names, return_tensors='pt', padding=True, truncation=True, max_length=16)
val_target_ids = tokenizer(val_sku_names, return_tensors='pt', padding=True, truncation=True, max_length=16)

# Prepare input tensors
train_input_ids = train_input_ids.input_ids.to(device)
val_input_ids = val_input_ids.input_ids.to(device)
train_attention_mask = (train_input_ids != tokenizer.pad_token_id).to(device)
val_attention_mask = (val_input_ids != tokenizer.pad_token_id).to(device)
train_target_ids = train_target_ids.input_ids.to(device)
val_target_ids = val_target_ids.input_ids.to(device)

# Use smaller batch size
batch_size = 16

# Define optimizer and loss function
learning_rate = 1e-6
regularization = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
smoothing_factor = 0.2
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=smoothing_factor)

# Train the BART model
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    for i in range(0, len(train_input_ids), batch_size):
        input_batch = train_input_ids[i:i + batch_size]
        attention_mask_batch = train_attention_mask[i:i + batch_size]
        target_batch = train_target_ids[i:i + batch_size]

        train_output = model(input_ids=input_batch, attention_mask=attention_mask_batch, labels=target_batch).logits
        train_loss = loss_fn(train_output.view(-1, train_output.shape[-1]), target_batch.view(-1))

        # Add regularization term to the loss
        l2_regularization = sum(p.pow(2.0).sum() for p in model.parameters())
        total_loss = train_loss + regularization * l2_regularization

        total_loss.backward()
        optimizer.step()

    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {total_loss.item()} - Time: {epoch_time:.2f} seconds")

# Save the trained model
# model_output_dir = 'C:/Users/Lenovo/Desktop/Web Scraping/bart_fixlrre_saved_model'
# model.save_pretrained(model_output_dir)
# print(f"Trained model saved to {model_output_dir}")

# Evaluate the BART model
model.eval()
val_output = model.generate(val_input_ids, attention_mask=val_attention_mask, max_length=16)
val_output_text = tokenizer.batch_decode(val_output, skip_special_tokens=True)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Model Name': val_model_names,
    'Predicted SKU Name': val_output_text,
    'True SKU Name': val_sku_names
})

# Add column indicating whether the prediction is correct
results_df['correct'] = results_df['Predicted SKU Name'] == results_df['True SKU Name']
results_df['correct'] = results_df['correct'].astype(int)

# Calculate and print accuracy
correct_predictions = (results_df['Predicted SKU Name'] == results_df['True SKU Name']).sum()
total_predictions = len(results_df)
accuracy = correct_predictions / total_predictions
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Export DataFrame to Excel
output_file = 'C:/Users/Lenovo/Desktop/Web Scraping/matched_tv_models_fixed_lr_reg_5.xlsx'
results_df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")

# Free up memory and cache
del results_df
torch.cuda.empty_cache()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the saved model (specify 'cuda' if available)
# loaded_model = BartForConditionalGeneration.from_pretrained("C:/Users/Lenovo/Desktop/Web Scraping/bart_fixlrre_saved_model").to(device)

# Tokenize the input model names from the new test dataset
new_test_df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/Sample Data_9.xlsx")
new_test_model_names = new_test_df["ORIGINAL MODEL NAME"].tolist()
new_test_input_ids = tokenizer(new_test_model_names, return_tensors='pt', padding=True, truncation=True, max_length=64)
new_test_input_ids = new_test_input_ids.input_ids.to(device)

# Free up GPU memory
torch.cuda.empty_cache()

# Generate predictions for the new test dataset
model.eval()
new_test_output = model.generate(new_test_input_ids, max_length=16)
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
new_test_output_file = 'C:/Users/Lenovo/Desktop/Web Scraping/new_test_data_with_predictions_5.xlsx'
new_test_df.to_excel(new_test_output_file, index=False)
print(f"New test data with predictions exported to {new_test_output_file}")

