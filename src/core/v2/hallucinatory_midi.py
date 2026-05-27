import numpy as np

class HallucinatoryMIDI:
    """Core V2: Translates neural errors into score data (MIDI)."""
    
    def extract_notes(self, flatness: np.ndarray, rms: np.ndarray, nan_rate: float = 0.0) -> list:
        """Converts Texture Metrics into a sequence of MIDI-like events.
        
        Logic:
        - High Flatness -> Higher Pitch (Neural screaming)
        - High RMS -> Velocity
        - NaN detected -> Sudden cluster or high-velocity 'Pain' note
        """
        notes = []
        for f, r in zip(flatness, rms):
            if r < 1e-4: continue
            
            # Map flatness [0.1, 0.6] to MIDI note [36, 96]
            pitch = int(36 + (f - 0.1) / 0.5 * 60)
            pitch = np.clip(pitch, 21, 108)
            
            # Map RMS to velocity [20, 127]
            velocity = int(20 + r * 107)
            velocity = np.clip(velocity, 0, 127)
            
            notes.append({"pitch": pitch, "velocity": velocity})
            
        if nan_rate > 0:
            # Inject chaotic 'Pain' clusters if NaN occurred
            for _ in range(5):
                notes.append({"pitch": 100, "velocity": 127, "meta": "nan_emergency"})
                
        return notes
