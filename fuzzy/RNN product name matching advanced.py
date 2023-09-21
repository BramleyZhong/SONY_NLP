import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Assuming df is your DataFrame containing "ORIGINAL MODEL NAME" and "KATABAN" columns
df = pd.read_excel("Sample Data_5.xlsx")
model_names = df["ORIGINAL MODEL NAME"].tolist()  # TV model names
sku_names = df["KATABAN"].tolist()  # Corresponding SKU names

# Tokenize the text data
model_tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
model_tokenizer.fit_on_texts(model_names)
model_total_words = len(model_tokenizer.word_index) + 1

sku_tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
sku_tokenizer.fit_on_texts(sku_names)
sku_total_words = len(sku_tokenizer.word_index) + 1

# Convert text to sequences of tokens
model_sequences = model_tokenizer.texts_to_sequences(model_names)
sku_sequences = sku_tokenizer.texts_to_sequences(sku_names)

max_model_sequence_length = max([len(seq) for seq in model_sequences])

# Pad sequences
padded_model_sequences = pad_sequences(model_sequences, maxlen=max_model_sequence_length, padding='post')
padded_sku_sequences = pad_sequences(sku_sequences, maxlen=max_model_sequence_length, padding='post')

# Create training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(padded_model_sequences, padded_sku_sequences, test_size=0.2, random_state=42)

# Calculate class weights for imbalanced data (if applicable)
class_counts = np.bincount(y_train.flatten())
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / num_samples for i, num_samples in enumerate(class_counts)}
class_weights_dict = dict(enumerate(class_weights))

# Build the RNN model
model = Sequential()
model.add(Embedding(model_total_words, 100, input_length=max_model_sequence_length))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(sku_total_words, activation='softmax'))  # Output layer with one neuron per SKU name token

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), class_weight=class_weights_dict)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", val_accuracy)

# Predict SKU names for validation data
predicted_sequences = model.predict(X_val)
predicted_sku_names = []
for seq in predicted_sequences:
    predicted_indices = [np.argmax(token_probs) for token_probs in seq]
    predicted_sku_name = sku_tokenizer.sequences_to_texts([predicted_indices])[0]
    predicted_sku_names.append(predicted_sku_name)

# Convert sequences back to texts for Model Name and True SKU Name columns
decoded_model_names = model_tokenizer.sequences_to_texts(X_val)
decoded_true_sku_names = sku_tokenizer.sequences_to_texts(y_val)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Model Name': decoded_model_names,
    'Predicted SKU Name': predicted_sku_names,
    'True SKU Name': decoded_true_sku_names
})

print(results_df)

# Export DataFrame to Excel
output_file = 'matched_tv_models00.xlsx'
results_df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")

# Print classification report
print(classification_report(decoded_true_sku_names, predicted_sku_names))
