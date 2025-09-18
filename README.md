# Agentic URL RAG (Gemini + Weaviate)

This project is a **Retrieval-Augmented Generation (RAG)** app built with:
- [Streamlit]for the UI
- [Gemini](via `agno`) as the LLM
- [Weaviate]as the vector database
- [Redis] for contextual memory(which I'm not able to use)
- [Agno]for agent + knowledge handling

It lets you load knowledge from URLs, store embeddings in Weaviate, and query them using Gemini.

---

## Features
- Load knowledge base from one or more URLs
- Store & search embeddings in **Weaviate**
- Chat with **Gemini 2.5 Flash** using RAG
- Simple **Streamlit UI** with sidebar for KB management

---

## Requirements

```bash
pip install -r requirements.txt
```

## Note
-  It's a timepass project built for learning about agno framework for building ai agents.
- Also how to use weaviate db(I dont like it much bcoz it will auto delete the cluster in a month)
- Also was trying to use redis for contextual memory(which was used locally coz college blocked redis cloud.Im not able to use in college wifi so)