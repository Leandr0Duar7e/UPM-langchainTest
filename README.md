# UPM-langchainTest
Demo for presentation about Langchain and LangGraph

## Quick Start

To get started with this project, follow these steps:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the `.env.template` file to `.env` and add your API keys:
   ```bash
   cp .env.template .env
   ```
   Then, open `.env` and replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key and add your `LANGSMITH_API_KEY`.

5. Launch LangSmith Studio by running the following command in the terminal:
   ```bash
   langgraph dev
   ```

6. Run the Jupyter notebooks:
   - For a demo of Langchain, open and run the `langchain-demo.ipynb` notebook in your IDE. (No need to run LangSmith Studio)
   - For a demo of LangGraph, open and run the `langgraph-demo.ipynb` notebook in your IDE.
