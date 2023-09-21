def find_substring_indices(main_string, substring):
    start_index = main_string.find(substring)
    end_index = start_index + len(substring)
    return (start_index, end_index, "MODEL")

import pandas as pd

def random_sample_dataframe(input_df, num_samples, random_state):
    # Use pandas sample function to randomly extract rows
    sampled_df = input_df.sample(n=num_samples, random_state=random_state)
    return sampled_df

dataframe = pd.read_excel("Sample Data.xlsx")
df = random_sample_dataframe(dataframe, num_samples=50, random_state=42)
training_data = []
for index, row in df.iterrows():
    training_data.append((row["ORIGINAL MODEL NAME"], {"entities": [find_substring_indices(row["ORIGINAL MODEL NAME"], row["KATABAN"])]}))

# training_data = [
#     ("Android Tivi Sony 4K 50 inch KD-50X80J/S", {"entities": [(12, 23, "MODEL")]})
#     # Add more annotated examples here...
# ]

import spacy
from spacy.training.example import Example

# Load the English language model
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.blank('en')

# Create a blank NER model
ner = nlp.get_pipe("ner")

# Add the entity label "MODEL" to the NER model
ner.add_label("MODEL")

# Convert training data to spaCy's Example format
examples = []
for text, annotations in training_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Training
optimizer = nlp.begin_training()
for _ in range(50):  # You can adjust the number of iterations
    losses = {}
    for example in examples:
        nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
    print("Losses:", losses)

# Save the trained model
nlp.to_disk("custom_ner_model")

# Load the custom trained NER model
custom_ner_model = spacy.load("custom_ner_model")

# Function to extract model numbers using the NER model
def extract_model_number(text):
    doc = custom_ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    model_numbers = [ent[0] for ent in entities if ent[1] == "MODEL"]
    return ", ".join(model_numbers)

dataframe["predicted model"] = dataframe["ORIGINAL MODEL NAME"].apply(extract_model_number)
dataframe.to_excel("predict.xlsx")

print(training_data)






