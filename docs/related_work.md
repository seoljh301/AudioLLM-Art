# AudioArt: Related Work Survey

## Overview

This document surveys sound art, music, and academic projects at the intersection of audio foundation models, neural audio codecs, and generative AI. The field encompasses latent perturbation (RAVE), token corruption (EnCodec/DAC), recursive audio↔text↔audio loops with AudioLLM captioners, and checkpoint morphing. This survey identifies foundational models, art installations, glitch traditions, tools, and research gaps that contextualize AudioArt's practice of *aesthetic misuse* of neural audio systems.

---

## Foundational Models & Papers

| Model | Authors / Lab | Paper / Repo | Key Use for AudioArt |
|-------|---------------|--------------|----------------------|
| **RAVE** | ACIDS-IRCAM | [arXiv:2111.05011](https://arxiv.org/pdf/2111.05011) / [GitHub](https://github.com/acids-ircam/RAVE) | Real-time VAE for latent space perturbation; raw latent manifold exploration and morphing |
| **EnCodec** | Meta AI | [arXiv:2210.13438](https://arxiv.org/pdf/2210.13438) / [GitHub](https://github.com/facebookresearch/encodec) | Neural audio codec with quantized latent codes; token corruption & databending directly on discrete codes |
| **DAC** | Descript | [GitHub](https://github.com/descriptinc/descript-audio-codec) | 90x compression, universal domain; Residual Vector Quantizer (RVQ) for multi-scale token manipulation |
| **SoundStream** | Google AI | [arXiv:2107.03312](https://arxiv.org/abs/2107.03312) / [Blog](https://ai.googleblog.com/2021/08/soundstream-end-to-end-neural-audio.html) | Real-time neural codec for speech+music; noise suppression + compression as artistic tool |
| **AudioLDM** | CVSSP | [arXiv:2301.12503](https://arxiv.org/pdf/2301.12503) / [GitHub](https://github.com/haoheliu/AudioLDM2) | Text-to-audio latent diffusion; loops with caption models for semantic drift |
| **AudioLDM 2** | CVSSP | [arXiv:2308.05734](https://arxiv.org/pdf/2308.05734) | Improved text-to-audio & text-to-speech; feed output back into captioner for recursive collapse |
| **MusicGen** | Meta AI / AudioCraft | [GitHub](https://github.com/facebookresearch/audiocraft) | Text & melody-conditioned generation; long-form musical material for checkpoint morphing |
| **Stable Audio** | Stability AI | [Blog](https://stability.ai/news/stable-audio-2-0) | Text-to-audio up to 3 minutes; audio-to-audio transforms via text prompts |
| **NSynth** | Google Brain / Magenta | [arXiv:1704.01279](https://arxiv.org/pdf/1704.01279) | WaveNet-based timbre interpolation; latent embedding space for smooth morphing |
| **Mimi Codec** | Kyutai Labs | [GitHub](https://github.com/kyutai-labs/moshi) / [Blog](https://kyutai.org/codec-explainer) | Streaming audio codec (12.5Hz, 1.1kbps); dual semantic+acoustic tokens for interpretation |
| **Qwen2-Audio** | Alibaba Cloud | [arXiv:2407.10759](https://arxiv.org/pdf/2407.10759) / [GitHub](https://github.com/QwenLM/Qwen2-Audio) | Multimodal audio-language model; native audio understanding for bidirectional loops |
| **SALMONN** | Tsinghua / ByteDance | [arXiv:2310.13289](https://arxiv.org/pdf/2310.13289) | Speech/audio/music language model; audio captioning for recursive caption→audio pipelines |
| **Pengi** | Microsoft | [arXiv:2305.11834](https://arxiv.org/abs/2305.11834) | Audio language model for open-ended tasks; audio captioning & QA for semantic loops |

---

## Sound Art & Installation Projects Using Generative Models

### Holly Herndon & Mat Dryhurst

- **Spawn** (2016–present): Neural network trained on Holly Herndon's voice; generates singing through collaborative human-AI ensemble. **Why:** Pioneering example of human-AI vocal co-creation and voice-space exploration; prototype for checkpoint morphing between trained models.

- **PROTO** (2019 album): Electronic pop choir mixing human and AI voices (Spawn); live performances with trained vocal ensemble. **Why:** Demonstrates long-form musical integration of neural synthesis and human performance; checkpoint blending in production workflow.

- **xhairymutantx** (2024 Whitney Biennial): Text-to-image AI trained on distorted self-portraits of Holly; explores identity and training data. **Why:** Meta-artistic treatment of model training; visual parallel to audio latent perturbation (training on corrupted input to shift aesthetic output).

- **The Call** / **Starmirror** (2023–2024): Interactive installations mixing AI choirs and audience participation. **Why:** Real-time latent interaction; audience co-creates generative output through voice input.

### Yuri Suzuki

- **Sonic Pendulum** (2017): 30 pendulums modulating AI-generated soundscape through doppler effect; ever-evolving. **Why:** Treats AI audio generation as live sculpture; latent space evolution in physical form; exemplar of token-level interpretation as gesture.

- **The Welcome Chorus** (Kent): Twelve horns representing districts, each singing AI-generated melody. **Why:** Spatializes generative audio; parallel to multi-track latent manipulation.

- **Vox PopulA.I** (Singapore Science Centre): Five towers generating real-time lyrics + melody via trained AI. **Why:** Interactive latent control; audience-driven model perturbation through voice input.

- **Arborhythm** (San Francisco, 2024): Environmental soundscape generation capturing urban/natural sounds. **Why:** Long-form ambient application of generative models; aesthetic databending of city audio.

### Dadabots (CJ Carr & Zack Zukowski)

- **24-hour Archspire stream** (2016–present): SampleRNN trained on death metal waveforms; continuous AI-generated extreme metal. **Why:** Raw waveform generation without music theory; embrace of neural artifacts as aesthetic; demonstrates long-training collapse into characteristic "Dadabots sound."

- **10+ generative albums** across metal/punk. **Why:** Genre-specific latent collapse; noise and distortion as feature, not bug; art from model degradation.

### Intelligent Instruments Lab (Iceland)

- **Notochord**: MIDI generation system for real-time performance & co-creation. **Why:** Discrete token (MIDI) manipulation; latent control interface for live performance.

- Live performances + workshops on human-AI musicianship. **Why:** Explores agency and authorship in human-neural systems; co-creativity as research.

### Mira Calix

- **"Nothing Is Set In Stone"** (2012 sculpture + sound): Monolithic musical sculpture at Natural History Museum. **Why:** Combines sculpture and generative sound; institutional recognition of AI sound art.

- **"Inside There Falls"** (2015, Sydney): 180-channel orchestral diffusion through 1.5km of hand-crushed paper installation. **Why:** Combines analog handmade aesthetic with digital; sound as spatial navigation.

### Robert Henke / Monolake

- Recent reflections on AI in music (2023–2024 interviews); critical engagement with generative tools in production. **Why:** Establishes skeptical artist perspective; questions context and authenticity; relevant for theoretical framing of "aesthetic misuse."

### Mat Dryhurst (Independent)

- Various AI ethics + art + copyright advocacy pieces; NFT sound art. **Why:** Frames artist intent around data provenance and ethical AI use.

---

## Glitch & Databending Tradition → Neural Codecs

### Historical Foundations

**Yasunao Tone** (1980s–90s): "Wounded" CDs created by scratching surfaces; indeterminate, unpredictable performances; challenges determinism in digital media.

**Oval** (1990s): German experimentalists who painted small images on CD undersides, causing intentional skipping. Transformed CD skipping into accessible glitch pop music; aesthetic of digital failure as compositional material.

**Nicolas Collins**: Pioneered circuit bending and live CD manipulation; influenced by both Tone and Oval.

### Connection to Neural Codecs

Traditional glitch art treated **corrupted digital data as instrument**. Neural audio codecs—RAVE, EnCodec, DAC—operate on **learned latent spaces and quantized tokens**, introducing new surfaces for corruption:

- **Latent perturbation** (RAVE): Modifying continuous latent vectors → smooth drift in timbre/texture (analog to Tone's indeterminacy).
- **Token databending** (EnCodec/DAC): Flipping bits in discrete codec tokens → introduces artifacts & reconstructions with characteristic neural texture (direct descendant of CD glitching, but learned instead of mechanical).
- **Recursive feedback** (AudioLLM loops): Audio→caption→audio chains degrade semantically (parallel to model "autophagy" in recent ML literature).

AudioArt extends this 30-year tradition: **codecs as instruments**, **corruption as composition**, **model collapse as aesthetic**.

---

## Open-Source Tools & Frameworks

| Tool | Purpose | Repo / Link | Use in AudioArt |
|------|---------|-------------|-----------------|
| **nn_tilde** | Max/MSP external for real-time neural model inference | [GitHub](https://github.com/acids-ircam/nn_tilde) | Live RAVE latent control in Max; real-time perturbation UI |
| **AudioCraft** | Meta's unified library (EnCodec, MusicGen, AudioGen, training code) | [GitHub](https://github.com/facebookresearch/audiocraft) | Training & fine-tuning codec models; baseline generation |
| **RAVE training repo** | Official ACIDS-IRCAM training code | [GitHub](https://github.com/acids-ircam/RAVE) | Train custom RAVE models on corrupted / prepared datasets |
| **Hugging Face Transformers** | Pre-trained Qwen2-Audio, Pengi, SALMONN integration | [Docs](https://huggingface.co/docs/transformers) | Easy loading of audio captioning models for recursive loops |
| **Diffusers** | Hugging Face diffusion library; AudioLDM2 pipeline | [GitHub](https://github.com/huggingface/diffusers) | Text-to-audio generation for audio→caption→audio chains |
| **torchaudio** | PyTorch audio I/O, resampling, feature extraction | [Docs](https://pytorch.org/audio) | Audio preprocessing; codec integration pipelines |
| **FastAI Audio** | High-level audio classification / fine-tuning | [Docs](https://docs.fast.ai) | Quick baseline models for audio understanding |
| **Audiocraft library** | Also includes MAT (Masked Audio Transformer) and other utilities | [Docs](https://audiocraft.metademolab.com/) | Extended generation & evaluation tools |
| **StreamingLLM / Moshi** | Real-time streaming codec + speech-text models | [GitHub](https://github.com/kyutai-labs/moshi) | Low-latency streaming audio for interactive loops |

---

## Recursive & Hallucinatory Pipelines: Papers & Concepts

### Model Collapse & Semantic Degradation

> The references below describe the *concept* of recursive collapse. Specific arXiv IDs in this section were not verified against the actual arXiv index and may have been hallucinated by the research pass — search by title before citing.

- **Shumailov et al., "The Curse of Recursion: Training on Generated Data Makes Models Forget"** (2024, [arXiv:2305.17493](https://arxiv.org/abs/2305.17493))
  - Coins "model collapse" — repeatedly training generative models on their own outputs causes catastrophic loss of distribution tails.
  - **Why:** Theoretical basis for what happens in MVP-B's recursive caption↔TTA loop after many iterations.

- **Alemohammad et al., "Self-Consuming Generative Models Go MAD"** (2024, [arXiv:2307.01850](https://arxiv.org/abs/2307.01850))
  - "Model Autophagy Disorder" — autoregressive loops on generated data show variance collapse and quality degradation.
  - **Why:** Provides the visual analogue (MAD images) of what MVP-B is doing in audio.

### Latent Space Exploration & Manipulation

- **Caillon & Esling, "RAVE: A variational autoencoder for fast and high-quality neural audio synthesis"** ([arXiv:2111.05011](https://arxiv.org/abs/2111.05011))
  - Original RAVE paper; describes the two-stage VAE-adversarial latent and the realtime traced export model.
  - **Why:** Mathematical basis for MVP-A latent perturbation.

- **Ainsworth et al., "Git Re-Basin: Merging Models modulo Permutation Symmetries"** (2023, [arXiv:2209.04836](https://arxiv.org/abs/2209.04836))
  - Aligns independently-trained nets via permutation before interpolation, recovering the linear-mode-connectivity property.
  - **Why:** Direct fix for MVP-D's collapse failure when morphing guitar↔organ — pre-align weights before interp.

---

## Conferences & Venues

### Academic Conferences

- **NIME** (New Interfaces for Musical Expression) — Annual, worldwide
  - Hosts papers on AI/ML in music, creative human-AI systems, sound art installations.
  - [nime.org](https://nime.org/) — See "Critical Perspectives on AI/ML in Musical Interfaces" and co-creativity papers.

- **ISMIR** (International Society for Music Information Retrieval) — Annual, worldwide
  - Dedicated "Creativity" track for generative models, artistic practice, human-AI co-creation.
  - [ismir.net](https://ismir.net/) — ISMIR 2026 includes "artistically-inspired generative tasks."

- **ICAD** (International Conference on Auditory Display) — Annual, worldwide
  - Sonification, interactive audio, auditory display; increasingly covers AI-driven audio.
  - [icad.org](https://icad.org/) — ICAD 2024 theme: "Sonification // Spatialization."

- **Audio Mostly** (biennial, Europe)
  - Interaction with sound; sonic design, product audio, emerging audio technologies.
  - [audiomostly.com](https://audiomostly.com/) — Accepts art + research + practice papers.

- **AIMC** (AI Music Creativity Conference) — Dedicated venue for AI in music composition, performance, and theory.

### Artist Residencies & Labs

- **ACIDS-IRCAM** (Paris) — Artistic research residencies; access to neural tools, RAVE, AudioCraft training.
  - [ircam.fr/creation](https://www.ircam.fr/creation/residence-en-recherche-artistique)

- **Intelligent Instruments Lab** (Reykjavík) — ERC-funded; AI in embodied instruments; mentorship + performance opportunities.
  - [iil.is](https://iil.is/)

- **Serpentine Galleries AI Lab** (London) — Hosts AI art residencies and exhibitions.

---

## Gaps: What AudioArt Fills

### Identified Lacunae in the Existing Landscape

1. **Explicit latent corruption as sound art material**
   - Much work explores caption→audio loops or timbral interpolation, but few projects systematically treat *latent token corruption* (bit-flipping in encoded audio) as live performance material.
   - AudioArt bridges glitch tradition + neural codecs explicitly.

2. **Checkpoint morphing across model architectures**
   - Papers study intra-model interpolation (Pengi, NSynth); no systematic art practice of blending checkpoints across *different models* (e.g., RAVE morph → EnCodec → MusicGen).
   - AudioArt makes cross-model morphing a primary aesthetic practice.

3. **Deliberate semantic collapse as compositional goal**
   - Literature treats model collapse as failure to be avoided; no art practice yet treats recursive audio→caption→audio loops as *intentional* degradation loops to generate new sonic spaces.
   - AudioArt embraces collapse as generative engine.

4. **Real-time latent interface design for audio**
   - RAVE + nn_tilde enable basic latent control; no mature, documented UI language for *sculpting* latent audio in live performance.
   - AudioArt will develop choreographed latent-space performance vocabulary.

5. **Aesthetic evaluation of codec artifacts**
   - Audio codecs are evaluated on fidelity (PESQ, MOS); artistic use of *characteristic codec artifacts* (e.g., EnCodec ringing, DAC timbre bias) not yet theorized.
   - AudioArt will catalog + celebrate codec "personalities."

6. **Institutional framing**
   - Holly Herndon, Yuri Suzuki, Dadabots have art-world recognition; no single research project yet centered on *AudioArt as a research + practice fusion*, with reproducible code + theory.
   - This project provides that.

---

## Key References

### Canonical Papers (Bookmarks)

1. [RAVE: arXiv:2111.05011](https://arxiv.org/abs/2111.05011)
2. [EnCodec: arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
3. [AudioLDM 2: arXiv:2308.05734](https://arxiv.org/abs/2308.05734)
4. [NSynth: arXiv:1704.01279](https://arxiv.org/abs/1704.01279)
5. [Shumailov et al., "The Curse of Recursion": arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
6. [Git Re-Basin: arXiv:2209.04836](https://arxiv.org/abs/2209.04836)

### Artists & Practitioners to Follow

- Holly Herndon & Mat Dryhurst
- Yuri Suzuki
- Dadabots (CJ Carr, Zack Zukowski)
- Mira Calix (deceased 2023, archival work)
- Robert Henke / Monolake
- Intelligent Instruments Lab (Halldór Úlfarsson et al.)

### Code Repos

- [acids-ircam/RAVE](https://github.com/acids-ircam/RAVE)
- [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- [acids-ircam/nn_tilde](https://github.com/acids-ircam/nn_tilde)
- [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)

---

**Last Updated:** May 2026  
**Scope:** Audio foundation models, neural codecs, generative audio, sound art + AI intersection  
**Intended Audience:** AudioArt researchers, sound artists, machine learning practitioners  
