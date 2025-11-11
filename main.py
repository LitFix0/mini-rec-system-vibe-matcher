# Install required packages if not installed:
# pip install --upgrade openai pandas scikit-learn matplotlib

import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import timeit

# Your provided OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Sample product data
products = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones dress perfect for outdoor festivals and summer vibes.", "vibes": ["boho", "relaxed"]},
    {"name": "Minimalist Blazer", "desc": "Structured, sharp lines for an urban professional look.", "vibes": ["minimal", "chic"]},
    {"name": "Athleisure Tracksuit", "desc": "Sporty, energetic tracksuit for workouts and street style.", "vibes": ["athleisure", "energetic"]},
    {"name": "Vintage Denim Jacket", "desc": "Classic blue denim, rugged with softly faded washes.", "vibes": ["vintage", "retro"]},
    {"name": "Cozy Knit Sweater", "desc": "Soft, oversized knit with warm autumn colors for cozy days.", "vibes": ["cozy", "casual"]},
    {"name": "Urban Cargo Pants", "desc": "Functional, edgy cargo pants designed for city adventure.", "vibes": ["urban", "edgy"]},
    {"name": "Elegant Silk Scarf", "desc": "Delicate silk scarf for a touch of elegance and color.", "vibes": ["elegant", "classic"]},
    {"name": "Techwear Windbreaker", "desc": "Sleek, waterproof windbreaker for futuristic streetwear.", "vibes": ["techwear", "futuristic"]}
]

df = pd.DataFrame(products)

# OpenAI embedding function using new API syntax
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

print("Generating embeddings for products...")
df['embedding'] = df['desc'].apply(get_embedding)

def vibe_match(user_query, k=3, threshold=0.7):
    query_embedding = np.array(get_embedding(user_query)).reshape(1, -1)
    item_embeddings = np.vstack(df['embedding'].values)
    sims = cosine_similarity(query_embedding, item_embeddings)[0]
    df['sim_score'] = sims
    top_matches = df.sort_values('sim_score', ascending=False).head(k)
    if top_matches.iloc[0]['sim_score'] < threshold:
        print("No strong matches found. Try a different vibe?")
        return top_matches[['name', 'desc', 'vibes', 'sim_score']]
    return top_matches[['name', 'desc', 'vibes', 'sim_score']]

# Example query
print(vibe_match("energetic urban chic"))

# Test with multiple queries and log latency
test_queries = ["energetic urban chic", "cozy fall outfit", "futuristic streetwear"]
metrics = []
item_embeddings = np.vstack(df['embedding'].values)

for q in test_queries:
    start = timeit.default_timer()
    query_vec = np.array(get_embedding(q)).reshape(1, -1)
    sims = cosine_similarity(query_vec, item_embeddings)[0]
    top_score = max(sims)
    elapsed = timeit.default_timer() - start
    metrics.append({"query": q, "good": top_score > 0.7, "latency": elapsed})

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

metrics_df.plot.bar(x='query', y='latency', legend=False)
plt.ylabel("Latency (seconds)")
plt.title("Query Latency")
plt.show()
