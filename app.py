import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the CSV
df = pd.read_csv("errors.csv")

# Load the SentenceTransformer model (this is the AI brain)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare your corpus of searchable texts (combine error columns for better matching)
corpus = df["Error Message"].tolist() + df["Cause"].tolist()
corpus_embeddings = model.encode(corpus)

# Initialize FastAPI
app = FastAPI()

@app.get("/search/")
def search_solution(query: str):
    # Convert the user query into an embedding (AI "understanding")
    query_embedding = model.encode([query])

    # Compute similarity between user query and the known error texts
    similarities = cosine_similarity(query_embedding, corpus_embeddings)

    # Find the best match
    best_match_index = np.argmax(similarities)

    # Decide if the match is from Error Message or Cause
    if best_match_index < len(df):
        row_index = best_match_index
    else:
        row_index = best_match_index - len(df)

    # Return the full result
    result = df.iloc[row_index]
    return JSONResponse({
        "matched_error_code": result["Error Code"],
        "error_message": result["Error Message"],
        "cause": result["Cause"],
        "resolution_steps": result["Resolution Steps"]
    })