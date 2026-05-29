import torch
import librosa
import logging
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

log = logging.getLogger("real_audio_judge")

class RealAudioJudge:
    """Uses Qwen2-Audio to listen to the generated wav and critique it."""
    
    def __init__(self, model_id="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
        self.device = device
        log.info(f"Loading AudioLLM Judge: {model_id} on {device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            log.info("AudioLLM Judge successfully loaded!")
        except Exception as e:
            log.error(f"Failed to load AudioLLM: {e}")
            raise

    def critique_audio(self, audio_path: str, target_aesthetic: str) -> str:
        """Listens to the audio file and returns a critique based on the target aesthetic."""
        log.info(f"Judge is listening to: {audio_path}")
        
        # Load audio (Qwen2-Audio expects 16kHz)
        audio, sr = librosa.load(audio_path, sr=16000)
        
        prompt = f"""
        <|audio_bos|><|AUDIO|><|audio_eos|>
        You are a harsh, avant-garde music critic. 
        The artist was trying to achieve this aesthetic: "{target_aesthetic}".
        Listen to the audio. 
        1. Describe exactly what rhythms and textures you hear.
        2. Give it a score from 1 to 10 based on how well it matches the target aesthetic.
        3. Give very specific instructions to the drum sequencer (e.g., "Add more high-frequency stuttering", "Make the bass drum hit more irregularly", "Increase the ratcheting on the snare") to improve it.
        Keep your response concise.
        """
        
        inputs = self.processor(
            text=prompt, 
            audios=audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        log.info(f"Judge Critique:\n{response}")
        return response
