import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load variables out of your local .env configuration file into memory
load_dotenv() 
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

# Load Data
reviews = pd.read_csv("data/womens_clothing_e-commerce_reviews.csv")
review_texts = reviews["Review Text"].dropna().reset_index(drop=True)
print(f"Number of reviews: {len(review_texts)}")

# Create Embeddings
response = client.embeddings.create(input=review_texts.tolist(), model=EMBEDDING_MODEL)
embeddings = [item.embedding for item in response.data]
print(f"Embeddings created: {len(embeddings)}")

# Dimensionality Reduction
embeddings_array = np.array(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_2d = tsne.fit_transform(embeddings_array)
print(f"2D embedding shape: {embeddings_2d.shape}")

# Visualize Embeddings
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, color='purple')
plt.title("2D Visualization of Review Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig("data/review_embeddings_plot.png")
print("Saved embedding visualization graph.")

# Feedback Categorization
categories = ["Quality", "Fit", "Style", "Comfort"]
category_descriptions = [
    "reviews discussing product quality, material quality, durability, stitching, fabric, craftsmanship",
    "reviews discussing size, sizing, fit, tightness, looseness, waist, length, measurements",
    "reviews discussing fashion style, design, appearance, trendiness, elegance, looks",
    "reviews discussing comfort, softness, coziness, ease of wearing, pleasant feeling"
]

category_response = client.embeddings.create(input=category_descriptions, model=EMBEDDING_MODEL)
category_embeddings = np.array([item.embedding for item in category_response.data])

def categorize_feedback(text_embedding, category_embeddings, categories):
    similarities = cosine_similarity([text_embedding], category_embeddings)[0]
    return categories[np.argmax(similarities)]

feedback_categories = [categorize_feedback(emb, category_embeddings, categories) for emb in embeddings]
reviews_clean = pd.DataFrame({"Review Text": review_texts, "Category": feedback_categories})
print("\nCategorized Sample Outputs:")
print(reviews_clean.head(5))

# Similarity Search
def find_similar_reviews(input_text, review_texts, embeddings, client, model, n=3):
    response = client.embeddings.create(input=[input_text], model=model)
    input_embedding = response.data[0].embedding
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-n:][::-1]
    return [review_texts.iloc[i] for i in top_indices]

example_review = "Absolutely wonderful - silky and sexy and comfortable"
most_similar_reviews = find_similar_reviews(example_review, review_texts, embeddings, client, EMBEDDING_MODEL, n=3)

print("\nMost Similar Reviews Matches:")
for i, review in enumerate(most_similar_reviews, start=1):
    print(f"{i}. {review}")
