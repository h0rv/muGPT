import csv

"""
kaggle datasets download -d eswarreddy12/family-guy-dialogues-with-various-lexicon-ratings
mkdir -p data
unzip family-guy-dialogues-with-various-lexicon-ratings.zip -d data
rm family-guy-dialogues-with-various-lexicon-ratings.zip
"""

file_path = "data/Family_Guy_Final_NRC_AFINN_BING.csv"


def get_corpus() -> str:
    corpus = []

    # Read dialogue and create the corpus
    with open(file_path, "r") as f:
        csvf = csv.DictReader(f, delimiter=",", quotechar="'")
        for row in csvf:
            dialogue = row["Dialogue"].strip('"')  # Remove double quotes
            corpus.append(dialogue)

    return "\n".join(corpus)
