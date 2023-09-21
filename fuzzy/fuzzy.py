from fuzzywuzzy import fuzz
import pandas as pd 

df = pd.read_excel("Sample Data_1.xlsx")

def match_tv_models(scraped_names, standard_names, threshold=50):
    matched_names = []

    for scraped_name in scraped_names:
        best_match = None
        best_score = 0

        for standard_name in standard_names:
            similarity_score = fuzz.partial_ratio(scraped_name, standard_name)
            
            if similarity_score > best_score and similarity_score >= threshold:
                best_score = similarity_score
                best_match = standard_name

        matched_names.append(best_match)

    return matched_names

# Example usage
#scraped_names = ["samsung uhd 4k", "lg oled 55", "sony bravia"]
scraped_names = df['ORIGINAL MODEL NAME'].tolist()
#standard_names = ["samsung uhd 4k tv", "lg oled 55-inch", "sony bravia smart tv"]
standard_names = df['KATABAN'].tolist()


matched_names = match_tv_models(scraped_names, standard_names)

#for scraped_name, matched_name in zip(scraped_names, matched_names):
    #print(f"Scraped: {scraped_name} => Matched: {matched_name}")

data = {'Scraped Name': scraped_names, 'Matched Name': matched_names, 'Standard Name': standard_names}
df = pd.DataFrame(data)

# Export DataFrame to Excel
output_file = 'matched_tv_models.xlsx'
df.to_excel(output_file, index=False)