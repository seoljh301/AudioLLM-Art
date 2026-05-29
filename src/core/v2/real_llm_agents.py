import os
import json
import logging
import torch
import re
from typing import List, Dict

log = logging.getLogger("real_llm_agents")

class RealTextComposerAgent:
    """Uses a powerful 7B Local LLM to dynamically generate high-complexity IDM sequences."""
    
    def __init__(self, model_id="Qwen/Qwen2-7B-Instruct", device="cuda"):
        # Upgraded to 7B model for better musical intelligence.
        self.device = device
        log.info(f"Loading Super-Intelligence LLM: {model_id}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto" # Auto-distribute across GPUs (e.g. CUDA 1,2,3)
            )
            self.model.eval()
            log.info("Super-Intelligence LLM successfully loaded!")
        except Exception as e:
            log.error(f"Failed to load LLM: {e}")
            raise

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _extract_json(self, text: str) -> str:
        """Extracts JSON array from text, handling markdown and slight truncations."""
        # 1. Clean markdown
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
            
        # 2. Extract array or object
        start_array = text.find('[')
        start_obj = text.find('{')
        
        start = start_array if start_array != -1 else start_obj
        if start != -1:
            text = text[start:]
            
        # 3. Attempt to parse, if it fails due to truncation, try to fix it
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, list):
                        parsed = v
                        break
            return json.dumps(parsed)
        except json.JSONDecodeError:
            last_brace = text.rfind('}')
            if last_brace != -1:
                text = text[:last_brace+1] + "\n]"
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if isinstance(v, list):
                                parsed = v
                                break
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    pass
        return "[]"

    def generate_palette(self, theme: str) -> str:
        """Asks the LLM to invent an audio palette."""
        sys_prompt = "You are an avant-garde sound designer. Output ONLY valid JSON array of sound objects. No explanations."
        user_prompt = f"""
        Invent 4 unique synthetic percussive sounds for this theme: "{theme}".
        Provide the output as a JSON array where each object has:
        - "symbol": a single uppercase letter
        - "description": a short phrase describing how it sounds
        - "duration": float between 0.1 and 0.4
        """
        response = self._generate_response(sys_prompt, user_prompt)
        json_str = self._extract_json(response)
        log.info(f"Generated Palette JSON:\n{json_str}")
        return json_str

    def generate_sequence(self, palette_json: str, theme: str, critique: str = "") -> str:
        """Asks the 7B LLM to perform 'Musical CoT' before outputting a complex JSON grid."""
        sys_prompt = "You are a professional IDM (Intelligent Dance Music) producer and tracker programmer. You specialize in complex, off-beat, and glitchy rhythms like Autechre or Aphex Twin."
        
        user_prompt = f"""
        Sound palette: {palette_json}
        Theme: {theme}
        {f'CRITIQUE OF YOUR PREVIOUS ATTEMPT: {critique}' if critique else ''}
        
        TASK: Composing a 1-bar DENSE and EVOLVING IDM rhythm.
        
        PROCESS (Musical Chain-of-Thought):
        1. Verbalize your rhythmic strategy. How will you use syncopation, ghost notes, and micro-timing to make it exciting and diverse?
        2. Output the final JSON array.
        
        JSON SCHEMA RULES:
        - "step": 0 to 31.
        - "sample": Use ONLY symbols from the palette.
        - "ratchet": 1 (normal) to 16 (extreme micro-stutter).
        - "velocity": 0.1 to 1.0 (loudness of this hit).
        - "offset_ms": -20 to 20 (pushes the hit forward or backward for 'wonky' groove).
        
        Place 18-25 events. Use complex groupings (e.g. 5-tuplets, 7-tuplets) by utilizing 'step' and 'offset_ms'.
        
        Output your thoughts first, then the JSON block.
        """
        response = self._generate_response(sys_prompt, user_prompt)
        log.info(f"LLM Musical Thought & Response:\n{response}")
        json_str = self._extract_json(response)
        return json_str
