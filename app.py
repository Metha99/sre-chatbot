import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Set page config
st.set_page_config(page_title="SRE Chatbot", page_icon="🤖", layout="centered")

# Title for the page
st.markdown("<h1 style='text-align: center;'>SRE Chatbot</h1>", unsafe_allow_html=True)

# Load model for semantic search (Optional, if you are searching queries based on similarity)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data from CSV files (adjust the paths to match your setup)
@st.cache_data
def load_data():
    # Load all the CSVs (adjust the filenames/paths accordingly)
    azure_df = pd.read_csv("azure_logs.csv")
    gitlab_df = pd.read_csv("gitlab_jobs.csv")
    servicenow_df = pd.read_csv("servicenow_tickets.csv")
    kb_df = pd.read_csv("knowledge_articles.csv")
    return azure_df, gitlab_df, servicenow_df, kb_df

# Filter data based on customer input
def filter_by_customer(customer_name):
    azure_df, gitlab_df, servicenow_df, kb_df = load_data()
    
    # Filter all dataframes by customer name
    azure_filtered = azure_df[azure_df['Customer'].str.contains(customer_name, case=False)]
    gitlab_filtered = gitlab_df[gitlab_df['Customer'].str.contains(customer_name, case=False)]
    servicenow_filtered = servicenow_df[servicenow_df['Customer'].str.contains(customer_name, case=False)]
    kb_filtered = kb_df[kb_df['Customer'].str.contains(customer_name, case=False)]
    
    return azure_filtered, gitlab_filtered, servicenow_filtered, kb_filtered

# Display results based on customer query
def display_customer_info(customer_name):
    azure_filtered, gitlab_filtered, servicenow_filtered, kb_filtered = filter_by_customer(customer_name)
    
    # Show Azure resource details
    if not azure_filtered.empty:
        st.markdown("### Azure Resources")
        for _, row in azure_filtered.iterrows():
            st.markdown(f"**Resource:** {row['Resource']}")
            st.markdown(f"**Status:** {row['Error']}")
            st.markdown(f"**OS:** {row['OS']}")
            st.markdown(f"**Subscription:** {row['Subscription_ID']}")
            st.markdown("---")
    else:
        st.markdown("### No Azure resources found for this customer.")
    
    # Show recent GitLab pipeline jobs
    if not gitlab_filtered.empty:
        st.markdown("### Recent GitLab Pipelines")
        for _, row in gitlab_filtered.iterrows():
            st.markdown(f"**Pipeline Name:** {row['Pipeline_Name']}")
            st.markdown(f"**Job Status:** {row['Status']}")
            st.markdown(f"**Pipeline URL:** {row['Pipeline_URL']}")
            st.markdown("---")
    else:
        st.markdown("### No recent GitLab pipelines found for this customer.")
    
    # Show open incidents and cases from ServiceNow
    if not servicenow_filtered.empty:
        st.markdown("### Open Incidents and Cases")
        for _, row in servicenow_filtered.iterrows():
            st.markdown(f"**Case ID:** {row['Case_ID']}")
            st.markdown(f"**Summary:** {row['Summary']}")
            st.markdown(f"**Related KB:** {row['Related_KB']}")
            st.markdown("---")
    else:
        st.markdown("### No open incidents or cases found for this customer.")
    
    # Show relevant Knowledge articles
    if not kb_filtered.empty:
        st.markdown("### Suggested Knowledge Articles")
        for _, row in kb_filtered.iterrows():
            st.markdown(f"**Article Link:** [{row['Article_Link']}]")
            st.markdown(f"**Summary:** {row['Summary']}")
            st.markdown("---")
    else:
        st.markdown("### No relevant Knowledge Articles found for this customer.")

# Input field for customer name
customer_name = st.text_input("Enter Customer Name:")

if customer_name:
    display_customer_info(customer_name)
