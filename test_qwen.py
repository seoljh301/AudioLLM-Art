import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

def test():
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    print("loading...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Generate 1 sec of silence/noise to test
    import numpy as np
    import soundfile as sf
    np.random.seed(0)
    audio_data = np.random.standard_normal(16000).astype(np.float32) * 0.1
    sf.write("test_audio.wav", audio_data, 16000)
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "test_audio.wav"},
            {"type": "text", "text": "What do you hear?"}
        ]}
    ]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print("Formatted text:\n", text)
    
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)[0])

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")
    
    print("generating...")
    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("RESPONSE:", response)

if __name__ == "__main__":
    test()
