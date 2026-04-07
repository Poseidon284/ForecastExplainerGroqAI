# ForecastExplainerGroqAI

ForecastExplainerGroqAI is a Streamlit-based frontend application that enables users to query documents and data using a retrieval-augmented chatbot. The application acts as an interface layer that forwards user input to a backend service and renders structured responses.

---

## Overview

This application is designed as a client layer for a document question-answering system. It does not implement retrieval or model inference internally. Instead, it:

- Collects user queries and file inputs
- Sends them to a backend API
- Displays responses returned by the backend

---

## Core Functionality

- Accepts natural language queries from users
- Supports document-based querying via file input
- Sends requests to a backend chatbot service
- Displays:
  - Generated answer
  - Supporting sources
- Maintains a direct mapping between backend response and UI

---

## System Flow
