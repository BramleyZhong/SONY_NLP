import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("Sample Data_5.xlsx")
#df = df[df['ORIGINAL MODEL NAME'].map(lambda x: x.isascii())]
df_test = pd.read_excel("new_test_data.xlsx")

'''
scraped_names = df_test['ORIGINAL MODEL NAME'].tolist()
standard_names = df_rest['KATABAN'].tolist()
true_names = df_test['KATABAN'].tolist()
'''

scraped_names = df_test['ORIGINAL MODEL NAME'].tolist()
standard_names = df['KATABAN'].tolist()
true_names = df_test['KATABAN'].tolist()


# Function to perform TV model name matching
def match_tv_models(scraped_names, standard_names, threshold=70, tfidf_threshold=0.70):
    matched_names = []

    # Tokenization and normalization for scraped names and standard names
    cleaned_scraped_names = [name.replace("TV", "").replace("Tv", "").replace("Speaker", "").replace("Headphone", "").replace("Bluetooth", "").replace("OLED", "").replace("LCD", "").replace("HD", "").replace("4K", "").replace("2K", "").replace("Samsung", "").replace("SAMSUNG", "").replace("AKXXT", "").replace("KXX", "").replace("smart", "").replace("Inch", "").replace("INCH", "").replace("inch", "").strip() for name in scraped_names]

    cleaned_standard_names = [name.replace("TV", "").replace("Tv", "").replace("Speaker", "").replace("Headphone", "").replace("Bluetooth", "").replace("OLED", "").replace("LCD", "").replace("HD", "").replace("4K", "").replace("2K", "").replace("Samsung", "").replace("SAMSUNG", "").replace("AKXXT", "").replace("KXX", "").replace("smart", "").replace("Inch", "").replace("INCH", "").replace("inch", "").strip() for name in standard_names]

    # Fuzzy matching
    for scraped_name in cleaned_scraped_names:
        best_match = None
        best_score = 0

        for standard_name in cleaned_standard_names:
            similarity_score = fuzz.partial_ratio(scraped_name, standard_name)

            if similarity_score > best_score and similarity_score >= threshold:
                best_score = similarity_score
                best_match = standard_name

        matched_names.append(best_match)

    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_standard_names)
    for i, scraped_name in enumerate(cleaned_scraped_names):
        scraped_vector = vectorizer.transform([scraped_name])
        cosine_similarities = cosine_similarity(tfidf_matrix, scraped_vector)
        matched_index = cosine_similarities.argmax()
        if cosine_similarities[matched_index] >= tfidf_threshold:
            matched_names[i] = standard_names[matched_index]

    return matched_names

# Perform TV model name matching
matched_names = match_tv_models(scraped_names, standard_names)

# Create a DataFrame
data = {'Scraped Name': scraped_names, 'Matched Name': matched_names, 'Standard Name': true_names}
df = pd.DataFrame(data)

# Export DataFrame to Excel
output_file = 'fuzzy_predict_on_new_test.xlsx'
df.to_excel(output_file, index=False)
print(f"Data exported to {output_file}")



