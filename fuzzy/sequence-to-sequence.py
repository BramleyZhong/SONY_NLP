import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet

# Assuming df is your DataFrame containing "ORIGINAL MODEL NAME" and "KATABAN" columns
df = pd.read_excel("Sample Data_4.xlsx")
df = df[df['ORIGINAL MODEL NAME'].map(lambda x: x.isascii())]

output_file = 'Sample Data_6.xlsx'
df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")


'''
model_names = df["ORIGINAL MODEL NAME"].tolist()  # TV model names
sku_names = df["KATABAN"].tolist()  # Corresponding SKU names

# Data Augmentation
augmented_model_names = []
for original_name in model_names:
    words = original_name.split()
    random.shuffle(words)
    augmented_words = []
    for word in words:
        if random.random() < 0.3:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms[0].lemma_names())
                augmented_words.append(synonym.replace("_", " "))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    augmented_name = " ".join(augmented_words)
    augmented_model_names.append(augmented_name)

all_model_names = model_names + augmented_model_names
all_sku_names = sku_names * 2

# Tokenize the combined data
model_tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
model_tokenizer.fit_on_texts(all_model_names)
model_total_words = len(model_tokenizer.word_index) + 1

sku_tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
sku_tokenizer.fit_on_texts(all_sku_names)
sku_total_words = len(sku_tokenizer.word_index) + 1

model_sequences = model_tokenizer.texts_to_sequences(all_model_names)
sku_sequences = sku_tokenizer.texts_to_sequences(all_sku_names)

max_model_sequence_length = max([len(seq) for seq in model_sequences])

padded_model_sequences = pad_sequences(model_sequences, maxlen=max_model_sequence_length, padding='post')
padded_sku_sequences = pad_sequences(sku_sequences, maxlen=max_model_sequence_length, padding='post')

X_train, X_val, y_train, y_val = train_test_split(padded_model_sequences, padded_sku_sequences, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential()
model.add(Embedding(model_total_words, 100, input_length=max_model_sequence_length))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(sku_total_words, activation='softmax'))  # Output layer with one neuron per SKU name token

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

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

decoded_model_names = model_tokenizer.sequences_to_texts(X_val)
decoded_true_sku_names = sku_tokenizer.sequences_to_texts(y_val)

results_df = pd.DataFrame({
    'Model Name': decoded_model_names,
    'Predicted SKU Name': predicted_sku_names,
    'True SKU Name': decoded_true_sku_names
})

print(results_df)

output_file = 'matched_tv_models_augmented01.xlsx'
results_df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")
'''