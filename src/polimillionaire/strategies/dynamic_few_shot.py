import time
import sqlite3
import json
import random
from collections import defaultdict
from polimillionaire.strategies.base import Strategy, Context, AnswerDecision
from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.strategies._common import build_decision, make_schema

class DynamicFewShotStrategy(Strategy):
    """Production-compliant K-Shot Strategy with pre-loaded memory banks."""
    
    def __init__(self, llm, db_path, k=3):
        self._llm = llm
        self.k = k
        self.strategy_name = f"dynamic_few_shot_k{k}"
        self.memory_bank = self._build_memory_bank(db_path)
        
    @property
    def model_name(self) -> str:
        return self._llm.name
        
    def _build_memory_bank(self, db_path):
        print(f"[{self.strategy_name}] Pre-computing Few-Shot memory bank...")
        bank = defaultdict(list)
        
        # Deterministic extraction. No GROUP BY hacks.
        query = """
            WITH Ranked AS (
                SELECT question_id, competition_id, question_text, options_json, correct_option_id_if_known,
                       ROW_NUMBER() OVER(PARTITION BY question_id ORDER BY id DESC) as rn
                FROM predictions
                WHERE correct_option_id_if_known IS NOT NULL
            )
            SELECT * FROM Ranked WHERE rn = 1
        """
        try:
            with sqlite3.connect(db_path) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute(query).fetchall()
                
            for row in rows:
                options = json.loads(row["options_json"])
                correct_text = next((opt["text"] for opt in options if opt["id"] == row["correct_option_id_if_known"]), "Unknown")
                
                bank[row["competition_id"]].append({
                    "id": row["question_id"],
                    "text": row["question_text"],
                    "correct_id": row["correct_option_id_if_known"],
                    "correct_text": correct_text
                })
        except sqlite3.OperationalError:
            print(f"[{self.strategy_name}] WARNING: Could not read {db_path}. Running as Zero-Shot.")
            
        return bank

    def __call__(self, question: Question, context: Context) -> AnswerDecision:
        start_time = time.perf_counter()
        
        domain_examples = self.memory_bank.get(context.competition_id, [])
        safe_examples = [ex for ex in domain_examples if ex["id"] != question.id]
        sampled_shots = random.sample(safe_examples, min(self.k, len(safe_examples)))
        
        system_content = "You are an expert answering multiple choice questions. Output valid JSON containing 'rationale', 'answer_id', and 'confidence'."
        user_content = "Here are some examples of correct reasoning:\n\n"
        
        for i, shot in enumerate(sampled_shots, 1):
            # [!] MANDATORY: Simulate the exact target JSON schema
            mock_json_response = {
                "rationale": f"The correct classification for this entity/concept is {shot['correct_text']}.",
                "answer_id": shot['correct_id'],
                "confidence": 1.0
            }
            user_content += f"Example {i}:\nQ: {shot['text']}\n"
            user_content += f"Assistant Output:\n{json.dumps(mock_json_response, indent=2)}\n\n"
            
        user_content += f"Now answer the following question:\nQ: {question.text}\n"
        for opt in question.options:
            user_content += f"- [{opt.id}] {opt.text}\n"
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        schema = make_schema(question, include_rationale=True)
        try:
            out = self._llm.complete_json(messages, schema)
        except Exception as e:
            out = {
                "rationale": f"JSON parsing failed: {e}",
                "answer_id": question.options[0].id,
                "confidence": 0.0
            }
            
        return build_decision(
            out,
            start_time,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=f"v2_json_fewshot_k{self.k}"
        )