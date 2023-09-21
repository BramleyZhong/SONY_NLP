import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW

# Load your DataFrame df containing "ORIGINAL MODEL NAME" and "KATABAN" columns
df = pd.read_excel("C:/Users/Lenovo/Desktop/Web Scraping/Sample Data_7.xlsx")
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
initial_lr = 1e-5
optimizer = AdamW(model.parameters(), lr=initial_lr)
smoothing_factor = 0.1
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean', label_smoothing=smoothing_factor)
# loss_fn = torch.nn.CrossEntropyLoss()

# Learning rate schedule
learning_rate_schedule = np.logspace(-6, -2, 10)
# [1.00000000e-06 2.78255940e-06 7.74263683e-06 2.15443469e-05 5.99484250e-05 1.66810054e-04 4.64158883e-04 1.29154967e-03 3.59381366e-03 1.00000000e-02]

# Regularization schedule
regularization_schedule = np.logspace(-6, -2, 10)

# Train the BART model with different learning rates and regularizations
num_epochs = 20
for lr_idx, learning_rate in enumerate(learning_rate_schedule):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for reg_idx, regularization in enumerate(regularization_schedule):
        print(f"Learning Rate: {learning_rate:.8f}, Regularization: {regularization:.8f}")

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

            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss.item()} - Time: {epoch_time:.2f} seconds")

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

        # Export DataFrame to Excel with different filename for each combination
        output_file = f'C:/Users/Lenovo/Desktop/Web Scraping/matched_tv_models_new_lr{lr_idx}_reg{reg_idx}.xlsx'
        results_df.to_excel(output_file, index=False)
        print(f"Data exported to {output_file}")

        # Free up memory and cache
        del results_df
        torch.cuda.empty_cache()

