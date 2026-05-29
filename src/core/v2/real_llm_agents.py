import os
import json
import logging
import torch
import re
from typing import List, Dict

log = logging.getLogger("real_llm_agents")

class RealTextComposerAgent:
    """Uses a local HuggingFace LLM to dynamically generate Palettes and Sequences."""
    
    def __init__(self, model_id="Qwen/Qwen1.5-1.8B-Chat", device="cuda"):
        # We use a smaller Qwen model (1.8B) so it easily fits in V100 VRAM alongside audio models.
        self.device = device
        log.info(f"Loading Local LLM: {model_id} on {device}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
            self.model.eval()
            log.info("Local LLM successfully loaded!")
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
            # Try to parse to see if it's already valid
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, list):
                        parsed = v
                        break
            return json.dumps(parsed)
        except json.JSONDecodeError:
            # It might be truncated. Find the last complete object.
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
        - "symbol": a single uppercase letter (e.g. "K", "S", "X")
        - "description": a short phrase describing how it sounds (e.g. "deep distorted kick", "glass shatter")
        - "duration": float between 0.1 and 0.4
        
        Example:
        [
          {{"symbol": "K", "description": "heavy metallic kick", "duration": 0.3}}
        ]
        """
        response = self._generate_response(sys_prompt, user_prompt)
        json_str = self._extract_json(response)
        log.info(f"Generated Palette JSON:\n{json_str}")
        return json_str

    def generate_sequence(self, palette_json: str, theme: str, critique: str = "") -> str:
        """Asks the LLM to sequence the palette into a 32-step grid."""
        sys_prompt = "You are a tracker sequencer programmer. Output ONLY valid JSON array. No explanations."
        
        user_prompt = f"""
        Here is your sound palette:
        {palette_json}
        
        Theme: {theme}
        {f'CRITIQUE OF PREVIOUS BEAT: {critique}' if critique else ''}
        
        Create a 1-bar drum pattern. 1 bar has 32 steps (0 to 31).
        Return a JSON array of events. Each event needs:
        - "step": integer from 0 to 31
        - "sample": the symbol of the sound (from the palette)
        - "ratchet": integer 1, 2, 3, 4, or 8 (1 = normal hit, >1 = extremely fast stutter/glitch in that step)
        
        Make it sound like modern IDM / Glitch music. Use ratcheting on some hi-frequency sounds.
        Output ONLY the JSON array.
        """
        response = self._generate_response(sys_prompt, user_prompt)
        log.info(f"Raw LLM Sequence Response:\n{response}")
        json_str = self._extract_json(response)
        log.info(f"Extracted Sequence JSON:\n{json_str}")
        return json_str
