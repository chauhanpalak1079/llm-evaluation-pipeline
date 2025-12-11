import json
import os
import time

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Pricing for GPT-4o-mini (approx. per 1k tokens) for cost estimation
        self.INPUT_COST_PER_1K = 0.00015
        self.OUTPUT_COST_PER_1K = 0.0006

    def load_json(self, file_path):
        """Helper to load JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None

    def estimate_tokens(self, text):
        """Simple approximation of token count (1 token ~= 4 chars)."""
        return len(text) / 4

    def calculate_cost(self, input_text, output_text):
        """Calculates estimated cost of the conversation turn."""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)

        cost = (
            ((input_tokens / 1000) * self.INPUT_COST_PER_1K) +
            ((output_tokens / 1000) * self.OUTPUT_COST_PER_1K)
        )

        return round(cost, 6), int(input_tokens + output_tokens)

    def evaluate_relevance(self, user_query, ai_response):
        """
        Asks the Judge LLM: Did the AI answer the user's question?
        """
        prompt = (
            f"You are an impartial expert evaluator. Your task is to grade the AI response "
            f"based on the User Query.\n\n"
            f"User Query: \"{user_query}\"\n"
            f"AI Response: \"{ai_response}\"\n\n"
            f"Evaluation Steps:\n"
            f"1. Identify the core intent of the user's query.\n"
            f"2. Check if the AI response addresses this intent directly.\n"
            f"3. Check for completeness: Did the AI miss any part of the question?\n"
            f"4. Assign a score based on the rubric below.\n\n"
            f"Scoring Rubric:\n"
            f"- 10: Perfect. Complete, direct, and helpful.\n"
            f"- 7-9: Good. Addresses main point but misses minor nuances.\n"
            f"- 4-6: Mediocre. Vague or partially incorrect.\n"
            f"- 1-3: Bad. Irrelevant or completely wrong.\n\n"
            f"Output strictly in JSON format: "
            f"{{ \"reasoning_steps\": \"<step-by-step logic>\", \"score\": <0-10> }}"
        )

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            return {"score": 0, "reasoning": f"Error: {str(e)}"}

    def evaluate_hallucination(self, ai_response, context_docs):
        """
        Asks the Judge LLM: Is the AI's answer supported by the context?
        """
        # Combine all vector chunks into one text block
        context_text = "\n".join([item['text'] for item in context_docs])
        # Truncate context to avoid token limits if necessary
        truncated_context = context_text[:10000]

        prompt = (
            f"You are a strict fact-checking AI. Your sole purpose is to verify if the "
            f"AI Response is grounded in the provided Source Context.\n\n"
            f"Source Context:\n{truncated_context}\n\n"
            f"AI Response:\n\"{ai_response}\"\n\n"
            f"Instructions:\n"
            f"1. Break the AI Response into individual claims.\n"
            f"2. For each claim, search the Source Context for evidence.\n"
            f"3. If a claim is NOT found in the context, it is a Hallucination.\n"
            f"4. Ignore external knowledge; only use the provided Context.\n\n"
            f"Output strictly in JSON format: "
            f"{{ "
            f"  \"status\": \"<PASS/FAIL>\", "
            f"  \"hallucinations\": ["
            f"      {{ \"claim\": \"<text of unsupported claim>\", \"reason\": \"<why it is not in context>\" }}"
            f"  ]"
            f"}}"
        )

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict fact-checker."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            return {"status": "ERROR", "unsupported_claims": [str(e)]}

    def run_pipeline(self, chat_file, context_file):
        print(f"--- Starting Evaluation for {chat_file} ---")
        start_time = time.time()

        # 1. Load Data
        chat_data = self.load_json(chat_file)
        vector_data = self.load_json(context_file)

        if not chat_data or not vector_data:
            return

        # 2. Extract specific turn (Last User Message + Last AI Message)
        messages = chat_data['fullContent']['conversation_turns']

        last_ai_msg = None
        last_user_msg = None

        # Iterate backwards to find the last AI message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'AI/Chatbot':
                last_ai_msg = messages[i]['message']
                # Find the user message immediately preceding this AI message
                for j in range(i - 1, -1, -1):
                    if messages[j]['role'] == 'User':
                        last_user_msg = messages[j]['message']
                        break
                break

        if not last_ai_msg or not last_user_msg:
            print("Could not find a valid User -> AI turn to evaluate.")
            return

        print(f"User Query: {last_user_msg[:50]}...")
        print(f"AI Response: {last_ai_msg[:50]}...")

        # 3. Get Context Documents
        context_docs = vector_data['fullContent']['data']['vector_data']

        # 4. Run Evaluations
        print("\n1. Evaluating Relevance...")
        relevance_result = self.evaluate_relevance(last_user_msg, last_ai_msg)
        print(f"Score: {relevance_result.get('score')}/10")
        print(f"Reason: {relevance_result.get('reasoning')}")

        print("\n2. Checking for Hallucinations...")
        hallucination_result = self.evaluate_hallucination(
            last_ai_msg, context_docs
        )
        print(f"Status: {hallucination_result.get('status')}")
        if hallucination_result.get('unsupported_claims'):
            print(f"Issues: {hallucination_result.get('unsupported_claims')}")

        # 5. Metrics
        eval_latency = time.time() - start_time
        cost, tokens = self.calculate_cost(last_user_msg, last_ai_msg)

        print("\n--- Final Metrics ---")
        print(f"Evaluation Latency: {eval_latency:.2f}s")
        print(f"Estimated Interaction Cost: ${cost:.6f}")
        print(f"Total Tokens Processed: {tokens}")
        print("-" * 30)


if __name__ == "__main__":
    # Ensure you have these files in your directory or update paths!
    evaluator = LLMEvaluator()
    evaluator.run_pipeline(
        "sample-chat-conversation-01.json",
        "sample_context_vectors-01.json"
    )