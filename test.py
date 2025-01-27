# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import spacy
from spacy.training.example import Example
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('loan_approval_dataset_1.csv')

# Preprocess data: Clean column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# Inspect column names
print(data.columns.tolist())

# Define intent and entity columns (update these based on actual column names)
intent_column = 'intent'  # Replace with actual intent column name
query_column = 'user_query'  # Replace with actual query column name
entity_columns = [col for col in data.columns if col not in [intent_column, query_column]]

# Prepare intent recognition data
queries = data[query_column].dropna().tolist()
intents = data[intent_column].dropna().tolist()
