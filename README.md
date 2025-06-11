# LLM-Powered Fact Checker
---

## Objective

Fact-checking system that takes in a news headline or social media post, extracts key claims, verifies them against a vector database of trusted facts, and classifies the result as:

*  **True**
*  **False**
*  **Unverifiable**

---

## Core Features

### 1. **Claim/Entity Detection**

* Extracts key claims/entities from text.
* Code: `core/claim_extractor.py`

### 2. **Trusted Fact Base**

* CSV file contain 30–50 verified facts from sources like PIB.
* Code: `data/trusted_facts.csv`

### 3. **Embedding + Vector Store**

* Embeds trusted facts using Sentence Transformers.
* Stores in FAISS index: `data/faiss_index.bin`
* Code: `build_database.py`, `core/vector_db.py`

### 4. **LLM-Powered Comparison**

* Extracted claims compared with top-k facts using an LLM.
* Code: `core/llm_service.py`, `pipeline.py`

### 5. **Streamlit UI**

* Lightweight frontend to test claims.
* Code: `app.py`

---

## Directory Structure

```
llm-fact-checker/
│
├── core/
│   ├── claim_extractor.py        # Extracts claims/entities from input
│   ├── llm_service.py            # LLM prompt generation and verdict classification
│   └── vector_db.py              # Embedding storage & retrieval (FAISS)
│
├── data/
│   ├── trusted_facts.csv         # Verified government facts
│   ├── pib_rss_feed.xml          # Raw press release feed
│   └── faiss_index.bin           # FAISS vector index of embedded facts
│
├── env/
│   └── .env                      # Environment variables (API keys etc.)
│
├── app.py                        # Streamlit frontend
├── build_database.py            # Script to build vector index
├── pipeline.py                  # Main pipeline for processing and verdicts
├── config.py                    # Config file for paths, models etc.
├── test_script.py               # script for quick testing
├── claim_extractor.log          # Logs
├── llm_service.log              # Logs
├── requirements.txt             # Dependencies
└── README.md                    
```

---

## Sample Input

**Claim:**

> "The Indian government has announced free electricity to all farmers starting July 2025."

**Output:**

```json
{
  "verdict": "False",
  "evidence": [
    "No official policy was found for July 2025 offering free electricity to all farmers."
  ],
  "reasoning": "The retrieved facts mention electricity reforms but no confirmation about free electricity to all farmers in July 2025."
}
```

---

## Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/llm-fact-checker.git
cd llm-fact-checker
```

2. **Create and activate virtual environment**

```bash
python -m venv env
source env/bin/activate       # For Linux/macOS
env\Scripts\activate          # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**
   Create a `.env` file inside `env/` folder:

```env
HUGGINGFACEHUB_API_KEY=your_huggingface_key
```

5. **Build the FAISS vector DB**

```bash
python build_database.py
```

6. **Run the Streamlit App**

```bash
streamlit run app.py
```

---

## Testing

You can test directly using:

```bash
python pipeline.py
```

Or use the UI via Streamlit.

---

<!-- ## 📸 Screenshots -->

<!-- --- -->