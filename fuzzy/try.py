from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm
import pandas as pd

def find_substring_indices(main_string, substring):
    start_index = main_string.find(substring)
    end_index = start_index + len(substring)
    return (start_index, end_index, "MODEL")


def random_sample_dataframe(input_df, num_samples, random_state):
    # Use pandas sample function to randomly extract rows
    sampled_df = input_df.sample(n=num_samples, random_state=random_state)
    return sampled_df

dataframe = pd.read_excel("Sample Data_2.xlsx")
df = random_sample_dataframe(dataframe, num_samples=2000, random_state=42)
training_data = []
for index, row in df.iterrows():
    training_data.append((row['ORIGINAL MODEL NAME'], {'entities': [find_substring_indices(row['ORIGINAL MODEL NAME'], row['KATABAN'])]}))

# training_data = [
#     ("Android Tivi Sony 4K 50 inch KD-50X80J/S", {"entities": [(12, 23, "MODEL")]})
#     # Add more annotated examples here...
# ]

print(training_data)

model = None
#output_dir=Path("C:\\Users\\nithi\\Documents\\ner")
n_iter=50
#dataframe["predicted model"] = dataframe["ORIGINAL MODEL NAME"].apply(extract_model_number)
#dataframe.to_excel("predict.xlsx")

#print(training_data)

#load the model

if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  
    print("Created blank 'en' model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

#for _, annotations in training_data:
    #for ent in annotations.get('entities'):
        #ner.add_label(ent[2])

ner.add_label("MODEL")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()


    examples = []
    for text, annots in training_data:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))
    nlp.initialize(lambda: examples)
    for itn in range(n_iter):
        losses = {}
        random.shuffle(examples)
        for batch in minibatch(examples, size=32):
            #texts, annotations = zip(*batch)
            #nlp.update(texts, annotations, sgd=optimizer, drop=0.2,losses=losses)
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        print(losses)
 

    #for itn in range(n_iter):
        #random.shuffle(training_data)
        #losses = {}
        #for text, annotations in tqdm(training_data):
            #nlp.update(
                #[text],  
                #[annotations],  
                #drop=0.5,  
                #sgd=optimizer,
                #losses=losses)
        #print(losses)


for text, _ in training_data:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

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
dataframe.to_excel("predict5.xlsx")

