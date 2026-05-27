# AudioArt

오디오 파운데이션 모델을 **의도적으로 오용**해 만드는 뉴럴 사운드 아트 프로토타입.

> AI가 소리를 어떻게 잘 이해하는가가 아니라, **AI가 소리를 이해하지 못할 때 소리가 어떻게 변형되는가**를 다룬다.

설계 철학: [`audio_foundation_model_sound_art_ideation.md`](audio_foundation_model_sound_art_ideation.md) ·
최종 작품 가이드: [`docs/META_SYMPHONY_ARCHITECTURE.md`](docs/META_SYMPHONY_ARCHITECTURE.md)

---

## 🎹 빠른 링크

- **🎧 데모 페이지**: [`demo.html`](demo.html) — 52 트랙 인라인 오디오 + waveform 썸네일 (브라우저 `file://` 또는 `python -m http.server`)
- **최종 결과물**: `runs/masterpiece/`
- **아키텍처 시각화**: [`docs/MULTINET_ARCHITECTURE.md`](docs/MULTINET_ARCHITECTURE.md)
- **구현 로그**: [`docs/IMPLEMENTATION_REPORT_V1.md`](docs/IMPLEMENTATION_REPORT_V1.md)
- **렌더 연대기**: [`docs/WORKFLOW_HISTORY.md`](docs/WORKFLOW_HISTORY.md)
- **관련 연구 서베이**: [`docs/Related Works/RESEARCH_SURVEY.md`](docs/Related%20Works/RESEARCH_SURVEY.md) + [`docs/Related Works/DETAILED_CASE_STUDIES.md`](docs/Related%20Works/DETAILED_CASE_STUDIES.md)
- **설계 원리 (한국어)**: [`docs/feedback_1.md`](docs/feedback_1.md)

---

## 🧩 9개 MVP 모듈

AudioArt 의 기본 단위는 9개의 독립적인 뉴럴 손상(neural damage) 연산자다.

| MVP | 도메인 | 백본 | 실패 모드 / 음상 |
|:---:|:---|:---|:---|
| **A** | Latent | RAVE | **Perturbation** — Brownian 노이즈 주입. 유기적 음색 떨림. |
| **B** | Semantic | Qwen2-Audio + AudioLDM2 *(또는 stub)* | **Caption Loop** — 오디오↔텍스트 재귀 번역. 논리적 표류. |
| **C** | Codec | EnCodec | **Token Bending** — bit-flip / shuffle / invalid-token. 디지털 알갱이. |
| **D** | Weight | RAVE × 2 | **Morphing** — state_dict 가중치 보간. 유령같은 붕괴. |
| **E** | Memory | RAVE | **Neural Granular** — 과거 latent 버퍼 투사. 시간이 번지는 코러스. |
| **F** | Spectral | RAVE | **Spectral Freeze** — 상위 latent 동결 + crossfade. Aurora shimmer. |
| **G** | Recursive | RAVE | **Latent Feedback** — Z-공간 delay-feedback. 자기 진화 메아리. |
| **H** | Generative | EnCodec | **Codebook Organ** — 소수/Fibonacci 토큰 합성. 추상적 드론. |
| **I** | Bass | EnCodec | **Bass Massive** — 하위 RVQ smear/jitter/fold. 심해 sub-bass. |

각 모듈은 `experiments/mvp_X_*/`에 독립 실험으로 격리되어 있고, `config.yaml` + `run.sh` + `main.py` + `results.json` 4종 세트를 가진다.

---

## 🌊 작곡 도구

단일 모듈 너머의 거시 작곡 인프라:

### 1. Multinet (`scripts/multinet.py`)
9개 MVP를 토폴로지로 묶는 매크로 네트워크 5종:

- **Net 1 — Crystal Cathedral** : 5-bus 병렬 공간 믹스
- **Net 2 — Recursive Organ** : 3-pass 매크로 피드백 루프
- **Net 3 — Decoding Chamber** : 9단 직선 손상 누적
- **Net Max — Cathedral Hive** : 8-bus 병렬 × 2-pass × cross-feedback (A–I 모두 사용)
- **Net Dynamic — Tempest** : 60초 타임라인 + 버스별 envelope + filter sweep + 3 impulse 이벤트

자세한 설계: [`docs/MULTINET_ARCHITECTURE.md`](docs/MULTINET_ARCHITECTURE.md)

### 2. Meta-Symphony (`scripts/meta_symphony.py`)
"Network of Networks". Net 1/2/3/Dynamic 결과를 3분 타임라인 위에서 LFO crossfade(60s/45s) + 스테레오 드리프트(20s/25s) + 100Hz sub 재주입으로 엮어낸다. AudioArt 스택에서 첫 stereo 출력물.

자세한 설계: [`docs/META_SYMPHONY_ARCHITECTURE.md`](docs/META_SYMPHONY_ARCHITECTURE.md)

### 3. 대용량 SoX 파이프라인
68분 이상의 1시간급 작품용 — Python 텐서 메모리가 병목이 될 때 `scripts/run_hfo_master_sox.sh` 등 SoX 기반 디스크 스트리밍으로 우회.

---

## 🛡️ 안전장치 — Texture Guard

모든 모듈은 청크별로 **Texture Governor**를 통과한다:

- **NaN Hardening** — 수치 폭주 감지 시 자동으로 wet=0 처리
- **Spectral Flatness 가드** — flatness > 0.55 → wet ×= 0.55 (백색 노이즈 회피)
- **RMS 가드** — silence (rms < 1e-4) 또는 clipping 임박 (rms > 0.85) 시 자동 감쇠
- **Centroid 가드** — 고역 fizz가 Nyquist × 0.42 초과 시 wet ×= 0.75

설계 원리(Anchored Corruption)는 [`docs/feedback_1.md`](docs/feedback_1.md) 참고.

---

## 🛠️ 설치

```bash
conda env create -f environment.yaml
conda activate audioart
# 대용량 SoX 파이프라인을 쓸 경우
conda install -y -c conda-forge ffmpeg sox
```

`torch==2.4.1 + cu121` 핀: V100 (SM 7.0) 지원. cu130 wheel은 SM ≥ 7.5 필요해 V100에서 silently fail.

### Meta-Symphony 렌더
```bash
PYTHONPATH=. python scripts/meta_symphony.py
```

### 데모 페이지
```bash
cd /home1/irteam/proj/AudioArt
python -m http.server 8765 --bind 127.0.0.1
# 브라우저: http://localhost:8765/demo.html
```

---

## 📁 디렉토리 구조

```
src/core/            Texture Governor, Mix Engine, OSC bridge, audio I/O, registry
src/modules/mvp_*/   9개 MVP 모듈 (A–I)
experiments/         MVP별 실험 + config + results.json
scripts/             Multinet, Meta-Symphony, mastering, SoX 래퍼, 데모 빌더
docs/                아키텍처 문서 + 워크플로우 + 관련 연구
runs/                생성된 wav · csv · log
checkpoints/         모델 가중치 (RAVE .ts, EnCodec)
demo.html            정적 데모 페이지
max/                 Max/MSP 패치 + OSC 포트 맵
```

---

## ✒️ 설계 근거 — Anchored Corruption

AudioLLM 과 codec 모델을 **불안정한 청각 기계**로 다룬다. 다만 그 "오용"이 노이즈 / 무음 / 클리핑 같은 비음악적 상태로 무너지지 않도록 **저음역 sub-bass crossover**와 **Texture Governor**로 닻을 내린다(anchor). 그 결과로 나오는 것은 단순한 글리치가 아니라, *모델이 만든 아름다운 오해* — 통제된 뉴럴 사운드 아트.

---

## 🔮 향후 작업 — AudioLLM 통합

프로젝트 이름의 **"AudioLLM-Art"** 가 가리키듯, 현재는 *뉴럴 사운드 아트* 단계지만 다음 단계에서는 진짜 AudioLLM을 본격적으로 도입한다.

- **MVP-B 실제 백본 활성화** — 현재 stub backend로만 동작. Qwen2-Audio-7B-Instruct (캡션) + AudioLDM2 (TTA) 다운로드(~18 GB) 후 실제 의미 표류 실험으로 확장.
- **AudioLLM 기반 새 MVP 제안군** — semantic-guided latent perturbation, prompt-conditioned token bending, multi-agent caption ensemble.
- **AudioLLM 자체 손상** — 모델의 latent semantic space에 직접 노이즈를 주입해 캡션 자체를 의도적으로 변질시키는 실험.
- **AudioLLM ↔ codec ↔ RAVE 크로스 도메인 체인** — 텍스트 손상 → 토큰 손상 → latent 손상이 한 그래프 안에서 함께 작동하는 통합 매크로넷.

이 단계로 넘어가면 프로젝트 이름이 비로소 정합적으로 작동한다.
