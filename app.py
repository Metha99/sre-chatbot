import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Unified AI Dashboard", layout="centered")

st.markdown("""
<style>
h1 { color: #9c27b0; text-align: center; }
.result-card {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    color: #e0e0e0;
    box-shadow: 0 0 10px rgba(156, 39, 176, 0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 Unified Incident Explorer</h1>", unsafe_allow_html=True)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    azure = pd.read_csv("azure_logs.csv")
    gitlab = pd.read_csv("gitlab_jobs.csv")
    snow = pd.read_csv("servicenow_tickets.csv")
    errors = pd.read_csv("errors.csv")

    # Create a combined dataframe on Incident_ID
    combined = pd.merge(errors, azure, on="Incident_ID", how="left")
    combined = pd.merge(combined, gitlab, on="Incident_ID", how="left")
    combined = pd.merge(combined, snow, on="Incident_ID", how="left")

    combined["embedding"] = combined["Error Message"].apply(lambda x: model.encode(str(x), convert_to_tensor=True))
    return combined

data = load_data()

# Search box
query = st.text_input("Ask about an incident, error, or keyword:")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_embedding, row)[0][0].item() for row in data["embedding"]]
    best_idx = scores.index(max(scores))
    result = data.iloc[best_idx]

    st.markdown("""
    <div class="result-card">
    <h3>🔍 Best Match Found</h3>
    <p><strong>Incident ID:</strong> {}</p>
    <p><strong>Case ID:</strong> {}</p>
    <p><strong>Customer:</strong> {}</p>
    <p><strong>Error Code:</strong> {}</p>
    <p><strong>Error Message:</strong> {}</p>
    <p><strong>Likely Cause:</strong> {}</p>
    <p><strong>Suggested Fix:</strong> {}</p>
    <hr/>
    <p><strong>Azure Resource:</strong> {} ({})</p>
    <p><strong>Subscription:</strong> {}</p>
    <p><strong>Azure Error:</strong> {}</p>
    <hr/>
    <p><strong>GitLab Job ID:</strong> {}</p>
    <p><strong>Pipeline:</strong> {} ({}) - <strong>Status:</strong> {}</p>
    <p><strong>Pipeline URL:</strong> <a href="{}" target="_blank">Open</a></p>
    <hr/>
    <p><strong>ServiceNow Summary:</strong> {}</p>
    <p><strong>Knowledge Base Ref:</strong> {}</p>
    </div>
    """.format(
        result['Incident_ID'],
        result.get('Case_ID', 'N/A'),
        result.get('Customer', 'N/A'),
        result['Error Code'],
        result['Error Message'],
        result['Cause'],
        result['Resolution Steps'],
        result.get('Resource', 'N/A'),
        result.get('OS', 'N/A'),
        result.get('Subscription_ID', 'N/A'),
        result.get('Error', 'N/A'),
        result.get('Job_ID', 'N/A'),
        result.get('Pipeline_Name', 'N/A'),
        result.get('Type', 'N/A'),
        result.get('Status', 'N/A'),
        result.get('Pipeline_URL', '#'),
        result.get('Summary', 'N/A'),
        result.get('Related_KB', 'N/A')
    ), unsafe_allow_html=True)
