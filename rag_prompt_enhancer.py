import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

class RAGPromptEnhancer:
    def __init__(self, attributes_csv, categories_csv, model_name="sentence-transformers/all-mpnet-base-v2"):
        # Load attributes and categories from CSVs
        self.attributes_df = pd.read_csv(attributes_csv)
        self.categories_df = pd.read_csv(categories_csv)
        self.model = SentenceTransformer(model_name)
        self.attributes_index = self.create_faiss_index(self.attributes_df, "name")
        self.categories_index = self.create_faiss_index(self.categories_df, "name")

    def create_embeddings(self, values: list):
        return np.array(self.model.encode(values))

    def create_faiss_index(self, df, column: str):
        embeddings = self.create_embeddings(df[column].tolist())
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def query_index(self, index, query, df, top_k=5):
        query_embedding = self.create_embeddings([query])
        _, indices = index.search(query_embedding, top_k)
        results = df.iloc[indices[0]]["name"].tolist()
        return results

    def enhance_prompt(self, prompt):
        attributes = self.query_index(self.attributes_index, prompt, self.attributes_df, top_k=5)
        categories = self.query_index(self.categories_index, prompt, self.categories_df, top_k=5)
        enhanced_prompt = f"Attributes: {', '.join(attributes)} | Categories: {', '.join(categories)}"
        return enhanced_prompt


if __name__ == "__main__":
    # Example usage
    attributes_csv = "attributes.csv"  # Path to the attributes CSV
    categories_csv = "categories.csv"  # Path to the categories CSV

    enhancer = RAGPromptEnhancer(attributes_csv, categories_csv)
    user_prompt = input("Enter your prompt: ")
    enhanced_prompt = enhancer.enhance_prompt(user_prompt)
    print(f"Enhanced Prompt: {enhanced_prompt}")