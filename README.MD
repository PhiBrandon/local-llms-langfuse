# Local LLMs with Logging
This repository is a simple example of langfuse with Ollama.

## Prereqs
Setup langfuse locally - https://langfuse.com/docs/deployment/local
Install Ollama - https://ollama.com/download
    - Download Qwen2 `ollama run qwen2` ** This requires 4.4GB of vram. If you need lower resources visit 
     `https://ollama.com/library/qwen2` to find the command for lower resource variants.
     ** You also need to update the code Line 33 with the new model name **


## Getting Started
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `touch .env`
    - Paste your `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` here
5. `python start.py`


Youtube guide: https://youtu.be/GkijOC_u1Xs