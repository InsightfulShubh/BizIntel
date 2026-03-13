# Project: Startup Intelligence Engine

## Project Goal

Build an AI-powered system that analyzes startup companies using Retrieval-Augmented Generation (RAG).

The system should allow users to:

- find startups similar to a given company
- perform startup competitor analysis
- generate SWOT analysis for a startup
- compare startups in the same domain
- explore startup ecosystems (e.g., AI infrastructure startups)

The system uses vector search and LLM-based reasoning.

---

# Data Sources

We use two open datasets downloaded from Kaggle.

## 1. Y Combinator Startup Dataset

Columns available:

company_id  
company_name  
short_description  
long_description  
batch  
status  
tags  
location  
country  
year_founded  
num_founders  
founders_names  
team_size  
website  
cb_url  
linkedin_url

This dataset contains YC startup descriptions.

---

## 2. Crunchbase Dataset

Main file used:

objects.csv

Important columns:

id  
entity_type  
name  
short_description  
description  
category_code  
tag_list  
country_code  
founded_at  
homepage_url  

The dataset contains many entity types, so we must filter:

entity_type == "Company"

---

# Data Processing Pipeline

Step 1 — Load datasets

Load YC dataset and Crunchbase dataset using pandas.

Step 2 — Filter Crunchbase companies

Only keep rows where:

entity_type == "Company"

Step 3 — Select useful fields

From Crunchbase:

name  
short_description  
description  
category_code  
tag_list  
country_code  
founded_at  

From YC:

company_name  
short_description  
long_description  
tags  
country  
year_founded  

Step 4 — Clean data

Remove rows where description is missing.

Remove duplicates.

Normalize fields into a common schema.

---

# Unified Startup Schema

Each startup will be converted to the following schema:

startup_id  
name  
description  
industry  
tags  
country  
founded_year  
source

Source will be either:

YC  
Crunchbase

---

# Document Generation for RAG

Each startup will be converted into a structured text document.

Example:

Company: Stripe  
Industry: Fintech  
Country: USA  
Founded: 2010  

Description:  
Stripe builds APIs and payment infrastructure for internet businesses.

Tags: payments, fintech, APIs

These documents will be embedded into a vector database.

---

# Vector Database

We will use:

FAISS or ChromaDB

Pipeline:

documents → embeddings → vector database

Embedding model options:

OpenAI embeddings  
Sentence Transformers

---

# RAG Pipeline

User Query
↓
Embedding generation
↓
Vector similarity search
↓
Retrieve top K startup documents
↓
LLM generates response using retrieved context

---

# Example Queries

Find startups similar to Stripe.

Explain competitors of OpenAI.

Generate SWOT analysis for Notion.

Compare YC startups building AI developer tools.

---

# Folder Structure

project_root/

data/
yc/
crunchbase/

processing/
data_loader.py
data_cleaning.py
document_builder.py

embeddings/
embedding_pipeline.py

vector_store/
vector_db.py

rag/
retriever.py
rag_pipeline.py

app/
api.py

---

# Development Approach

We will build the project in stages.

Stage 1:
Load and clean YC + Crunchbase datasets.

Stage 2:
Generate RAG-ready documents.

Stage 3:
Create embeddings and vector database.

Stage 4:
Build semantic startup search.

Stage 5:
Add AI analysis features (SWOT, comparison).

---

# Coding Guidelines

Use Python.

Use pandas for data processing.

Write modular code.

Each stage should be implemented in separate modules.

Avoid large monolithic scripts.

Add clear docstrings to each function.
