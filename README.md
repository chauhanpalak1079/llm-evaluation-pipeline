# LLM Evaluation Pipeline

An automated Python-based pipeline designed to evaluate the quality of AI chatbot responses. This tool assesses interactions based on relevance, factual accuracy (hallucination), and estimated cost/latency using a "Judge LLM" approach.

## ⚠️ Important: API Key Required
**This tool utilizes the OpenAI API to perform evaluations.**
To run this pipeline, you must provide your own OpenAI API key. The script does not come with a pre-configured key.
(See the [Setup](#local-setup--installation) section below for instructions).

## 📊 Evaluation Metrics
The pipeline tests AI answers against three key parameters:
1.  **Response Relevance:** Does the answer directly address the user's intent? (Scored 0-10)
2.  **Hallucination Check:** Is the answer fully supported by the provided context vectors? (Pass/Fail)
3.  **Latency & Cost:** Estimates the token usage and cost for the interaction (based on GPT-4o-mini pricing).

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/chauhanpalak1079/llm-evaluation-pipeline.git](https://github.com/chauhanpalak1079/llm-evaluation-pipeline.git)
cd llm-evaluation-pipeline
````

### 2\. Set up Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install openai python-dotenv
```

### 4\. Configure Environment Variables

Create a file named `.env` in the root directory. Add your OpenAI API Key inside:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

*Note: Without this key, the script will fail to authenticate.*

## 🏃 Usage

Ensure you have your input JSON files (`sample-chat-conversation-01.json` and `sample_context_vectors-01.json`) in the root directory.

Run the evaluation script:

```bash
python eval_pipeline.py
```

### Sample Output

```text
--- Starting Evaluation ---
User Query: "What is the cost of IVF?"
AI Response: "The cost is approx Rs 3,00,000..."

1. Evaluating Relevance...
Score: 10/10
Reason: The response provides specific numbers directly addressing the user's question about cost.

2. Checking for Hallucinations...
Status: PASS

--- Final Metrics ---
Evaluation Latency: 1.42s
Estimated Interaction Cost: $0.000120
Total Tokens Processed: 340
```

## 🏗 Architecture & Design Decisions

### Architecture

The pipeline follows a **"Model-Based Evaluation" (Judge LLM)** architecture:

1.  **Ingestion:** Parses raw chat logs and retrieved context chunks from JSON files.
2.  **Prompt Engineering:** Constructs specific evaluation prompts (Chain-of-Thought for relevance, Citational Verification for hallucinations).
3.  **Judge Execution:** Sends these prompts to a highly capable but efficient model (GPT-4o-mini) to act as the grader.
4.  **Reporting:** Aggregates scores, detects failures, and calculates operational metrics.

### Why this approach?

  * **Decoupled Logic:** We separate the *evaluation* logic from the *generation* logic. This mimics a real-world production setup where logs are analyzed asynchronously.
  * **Judge LLM:** Using an LLM to evaluate another LLM is currently the industry standard for unstructured text data where simple regex or keyword matching fails.
  * **GPT-4o-mini:** We chose this model as the judge because it offers the best balance of reasoning capability and low cost, making it feasible for frequent testing.

## 📈 Scalability Strategy

**Question:** If we run this script at scale (millions of daily conversations), how do we ensure low latency and costs?

Running a "Judge LLM" on every single message in real-time is prohibitively expensive and slow. To scale this to millions of users, we would implement a **Tiered Sampling Architecture**:

1.  **Tier 1: Real-Time Guardrails (100% of traffic)**

      * **Method:** Lightweight, non-LLM checks (Regex, Keyword blocklists, PII detection).
      * **Cost:** Negligible.
      * **Latency:** \<20ms.

2.  **Tier 2: Async Sampling (1-5% of traffic)**

      * **Method:** Instead of evaluating every chat, we randomly sample 1-5% of conversations and push them to an **Async Message Queue** (e.g., Kafka or Celery).
      * **Processing:** Worker nodes pick up these logs and run the `eval_pipeline.py` script in the background. This ensures the user's actual chat experience is never slowed down by the evaluation process.

3.  **Tier 3: Targeted "Red Flag" Eval**

      * **Method:** If a user leaves negative feedback (thumbs down) or the conversation length exceeds a certain threshold, we force-trigger an evaluation for that specific chat to diagnose the issue.

4.  **Cost Optimization:**

      * **Batching:** Send requests to the Judge LLM in batches to reduce network overhead.
      * **SLM (Small Language Models):** For simple relevance checks, fine-tune a smaller, cheaper local model (like Llama-3-8B) to replace GPT-4, reducing API costs to near zero.

<!-- end list -->

```
```