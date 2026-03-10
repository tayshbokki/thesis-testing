NAIVE RAG BASELINE — DLSU CpE Checklist AY 2022-2023
Dense Retrieval (all-MiniLM-L6-v2) + ChromaDB + LLM via HF Inference API

- Hard-coded knowledge base (no database yet)
- Proves RAG pipeline works before adding DB complexity
- Establishes ROUGE-L, response time baseline scores, more testing basis to add
when llama is available


## Setup
1. Clone the repo
2. change `.env.example` to `.env`
3. Generate your HF token at huggingface.co/settings/tokens
4. Paste your token into `.env`
5. pip install -r requirements.txt
```
