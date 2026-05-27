# AudioArt Implementation Report V1 (2026-05-22)

## 1. 개요
본 문서는 `feedback_1.md`에 제시된 "Anchored Corruption" 및 "Texture Guard" 원칙을 바탕으로 AudioArt 시스템에 적용된 모든 기술적 변경 사항과 실험 결과물을 누적하여 기록한다.

---

## 2. 핵심 아키텍처 (`src/core/`)

### 2.1 `mix.py` (Mixing Engine)
*   **Dry/Wet Anchoring**: 원본(Dry)과 변조(Wet) 신호의 비율 제어.
*   **RMS Matching**: 신호 간 에너지를 일치시켜 급격한 볼륨 변화 방지.
*   **Low-End Protection (Crossover)**: 특정 주파수(예: 80Hz, 150Hz) 이하의 원본 신호를 보존 및 부스트하는 로직 구현.
*   **Safety**: `np.nan_to_num` 및 Tanh Soft Limiter를 통한 수치적 안정성 확보.

### 2.2 `texture_metrics.py` (Real-time Analysis)
*   **Metrics**: Spectral Flatness, Spectral Centroid, RMS, Zero Crossing Rate 분석.
*   **NaN Detection**: 모델 폭주로 인한 수치 오류(NaN, Inf) 실시간 감지.

### 2.3 `texture_governor.py` (Automatic Guardrail)
*   **Governor**: 분석된 지표가 임계치를 넘을 경우 실시간으로 `wet` 비율을 감쇠시켜 사운드 붕괴 방지.
*   **Emergency Stop**: NaN 감지 시 즉시 Wet 신호를 0.0으로 차단.

---

## 3. MVP 모듈 강화

### 3.1 MVP-A (Latent Perturbation)
*   **Smoothed Noise**: 단순 White Noise 대신 브라운 운동 방식의 Smoothed Drift 적용. 유기적인 음색 변화 유도.

### 3.2 MVP-C (Token Bending)
*   **Local Shuffle Window**: 시간 축 전체가 아닌 미세 윈도우 내에서만 토큰을 섞어 리듬 구조 보존.
*   **Auto-Upper Selection**: 하위 퀀타이저(Body)는 보호하고 상위 퀀타이저(Detail)만 변조.

### 3.3 MVP-D (Checkpoint Morphing)
*   **Ghost Layer Synthesis**: 모델 붕괴 지점을 얇은 배경 레이어로 활용하여 유령 같은 잔향 질감 생성.

### 3.4 MVP-E (Neural Latent Granular) - *NEW*
*   **Latent Memory**: 최근 40초간의 잠재 벡터를 버퍼에 저장.
*   **Neural Graining**: 과거의 잠재 조각들을 현재에 무작위로 투사하여 "시간이 번진" 듯한 질감 구현.

### 3.5 MVP-F (Neural Spectral Frozen) - *NEW*
*   **Stochastic Shimmer**: 상위 50%의 층위(음색 광택)만 특정 시점의 값으로 동결. 
*   **Periodic Crossfade**: 128 프레임 주기로 동결 상태를 부드럽게 갱신하여 인위적인 끊김(Beep) 대신 몽환적인 '일렁임(Shimmer)' 창출.

### 3.6 MVP-G (Latent Feedback Echo) - *NEW*
*   **Neural Feedback Loop**: 오디오가 아닌 잠재 공간 내에서 `z_out = z_curr + z_delay * feedback` 루프 형성.
*   **Evolving Echo**: 단순 반복이 아닌, 모델이 해석을 거듭하며 음색 자체가 기괴하게 진화하는 뉴럴 피드백.

### 3.7 MVP-H (Codebook Organ) - *NEW*
*   **Generative Engine**: 외부 입력 오디오 없이, 소수(Prime)나 피보나치 수열을 이용해 NAC의 코드북 인덱스를 직접 생성.
*   **Abstract Sound**: 인간의 개입이 완전히 배제된 신경망의 "원시적 언어" 연주.

### 3.8 MVP-I (Neural Bass Massive) - *NEW*
*   **Low-End Modulations**: EnCodec의 하위 퀀타이저(0~2) 레이어만 타겟팅.
*   **Temporal Smearing & Codebook Jitter**: 저음의 시간을 늘려 심해 같은 공간감을 주고, 인접 코드북 인덱스를 섞어 거친 '뉴럴 디스토션' 생성.

---

## 4. [심화] 수치적 안전장치: Texture Governor & NaN Hardening
3분 이상의 롱폼 렌더링에서 발생하는 예외 상황을 완벽히 통제하기 위한 기술적 명세이다.

### 4.1 실시간 지표 임계값 (Metric Thresholds)
*   **Spectral Flatness (0.60)**: 소리의 무작위성을 측정. 0.6 이상일 경우 '음악적 질감'을 상실한 것으로 간주하여 `wet_ratio *= 0.55` 적용.
*   **Spectral Centroid (Nyquist * 0.45)**: 소리의 밝기를 측정. 고역대 노이즈(Fizz)가 과도할 경우 `wet_ratio *= 0.75` 적용.
*   **RMS Min/Max (1e-4 / 0.90)**: 소리의 실종(Collapse) 및 클리핑(Clipping) 방지.

### 4.2 NaN Emergency 처리 로직
*   **문제**: 다중 신경망 체인(A->D->C 등)에서 가중치 임계값 충돌로 인해 `NaN(Not a Number)`이 발생, 특정 시점(예: 2분 8초)에서 오디오가 끊기는 현상.
*   **해결**: `compute_texture_metrics`에서 NaN 실시간 체크. 발견 시 해당 청크의 변조 신호를 0으로 강제 치환하고 원본(Dry)만 출력하여 **재생의 연속성**을 확보함.

---

## 5. [심화] 주파수 분리 및 보강: Sub-Bass Anchoring
사용자 요청에 따른 "단단한 저음"을 위해 도입된 주파수 분할 방식이다.

### 5.1 2차 버터워스 교차 필터 (2nd-order Butterworth Crossover)
*   **Cutoff**: 80Hz (서브우퍼 대역 타겟팅).
*   **Slope**: 12dB/oct. 
*   **작동**: 80Hz 이하의 신호는 모든 변조 파이프라인을 우회하여 최종 출력단으로 직결됨.

### 5.2 가중치 부스트 (Sub-Boost)
*   **Gain**: +8.0dB. 
*   **이유**: 다단계 변조 과정에서 발생하는 위상 상쇄와 에너지 손실을 보상하기 위해 앵커링된 저음에 추가 게인을 부여하여 타격감을 극대화함.

---

## 6. 최종 마스터링 표준: LUFS Normalization
*   **Standard**: BS.1770-4 (국제 방송/스트리밍 규격).
*   **Target Loudness**: **-12.0 LUFS**.
*   **True Peak**: -0.5 dBTP.
*   **Tool**: `pyloudnorm` 오픈소스 라이브러리 활용.

---

## 7. 대용량 오디오 마스터링 최적화 (OOM Bypass)

### 7.1 SoX (Sound eXchange) 커맨드라인 프로세싱
*   **문제**: `librosa`를 이용한 실시간 리샘플링과 메모리 상의 거대 행렬(Float32, 68분, 8채널) 연산이 시스템 RAM을 100% 점유하여 강제 종료됨.
*   **해결**: 파이썬의 오디오 연산 의존도를 낮추고, C 기반의 최적화된 오디오 프로세싱 도구인 **SoX**를 도입.
*   **구현**: 디스크 스트리밍 방식의 필터링(`sinc`), 피치 시프팅(`pitch`), 멀티트랙 믹싱(`-m`)을 백그라운드 쉘 스크립트로 분리하여 실행.
*   **결과**: 메모리 누수 없이 68분짜리 오디오 8개를 10여 분 만에 안정적으로 합성에 성공.

### 7.2 복합 체인 동기화: Phase 2 Min-Length Cropping
*   **문제**: 여러 네트워크(Net 1~Dynamic)를 결합할 때, EnCodec의 스트라이드 패딩 문제로 인해 트랙 간에 수십 밀리초의 길이 차이가 발생하여 배열 연산 오류(`ValueError: operands could not be broadcast`)가 발생함.
*   **해결**: `meta_symphony.py`에 가장 짧은 트랙 길이에 맞춰 모든 레이어와 LFO 배열을 실시간으로 자르는(Crop) 동기화 로직을 추가하여 3분 이상의 복합 믹싱 안정성을 확보함.

---

## 8. 최종 마스터피스 및 마스터링 기법 명세

### 8.1 Multi-source Granular Masterpiece (`neural_granular_symphony.wav`)
*   **기법**: 8개의 소스 트랙을 250ms 단위의 잘게 쪼갠 조각(Grain)으로 분리.
*   **Stochastic Interweaving**: 매 순간 무작위로 3개의 트랙만 선택하여 재생.
*   **Micro-Temporal Jitter**: 선택된 조각들에 +/- 50ms의 시간적 오차를 두어 거대한 코러스 효과 창출.
*   **결과**: 정적인 레이어링을 벗어나, 소리들이 서로 뒤섞이며 폭풍우처럼 몰아치는 역동적인 "신경망 소용돌이" 생성.

### 8.2 Full-Spectrum Octave Spreading (`full_spectrum_neural_galaxy.wav`)
*   **기법**: 8개 트랙을 단일 피치가 아닌, **가청 주파수 전체(-2 옥타브 ~ +2 옥타브)로 강제 분산 배치**.
*   **Volume Balancing**: 저음역대(Sub)는 볼륨을 키우고 초고음역대(Sparkle)는 볼륨을 줄이는 삼각 형태의 에너지 밸런싱.
*   **결과**: 가청 주파수를 빈틈없이 채우는 초거대 '뉴럴 하이퍼-코드(Hyper-Chord)' 완성.

### 8.3 High-Fidelity Enhancement (`hifi_enhanced_symphony_1.wav`)
*   **기법**: 저음역 간섭 배제, 초고음역대 해상도 극대화.
*   **Harmonic Exciter**: 8kHz 이상 대역에 비선형 포화(Non-linear saturation)를 걸어 새로운 배음을 인공적으로 생성.
*   **Air Boost**: 14kHz 대역 +10dB 쉘프 부스트 적용.
*   **결과**: 답답한 디지털 질감을 벗어나, 크리스탈처럼 맑고 비싼(Expensive) 느낌을 주는 HD급 사운드 텍스쳐.

### 8.4 Foundation 9-Octave Hyper-Chord (`foundation_hyperchord_limited.wav`)
*   **기법**: 오직 원본(Foundation) 트랙 하나만을 재료로 사용하여 -4 옥타브부터 +4 옥타브까지 총 9번 겹쳐 쌓음.
*   **결과**: 신경망 변조 없이 순수 피치 시프팅만으로 구축된, 파이프 오르간을 연상케 하는 웅장한 아날로그 드론 사운드.

### 8.5 Earthquake Bass: Extreme Low-End EQ (`foundation_hyperchord_EARTHQUAKE.wav`)
*   **기법**: 9옥타브 중첩 과정에서 발생하는 중음역대 마스킹 현상을 해결하기 위한 극단적 보정.
*   **서브우퍼 타겟팅**: 50Hz 대역에 **+15dB**의 초강력 부스트 적용.
*   **Mud Scoop**: 300Hz 대역을 -6dB 깎아내어 저음의 투명도 확보.
*   **결과**: 단순한 볼륨 증가를 넘어, 공간 전체를 물리적으로 진동시키는 압도적인 서브 베이스 질감 구현.

---

## 9. 결과물 위치 및 최종본
*   **최종 작품**: `runs/masterpiece/final_symphony_bass_heavy.wav`
*   **안정성**: 180초 전 구간 결함 없음.
*   **질감**: 8개 레이어의 유기적 중첩 및 LFO 오토메이션.
*   **에너지**: -12 LUFS의 높은 음압과 +8dB의 서브 베이스 타격감.

---

## 10. Multinet — 매크로 네트워크 5종 (`scripts/multinet.py`)

`feedback_1.md` 의 anchored corruption 원리를 단일 MVP 너머의 거시 작곡에 적용한 5개의 매크로 신호 그래프. 자세한 토폴로지는 [`MULTINET_ARCHITECTURE.md`](MULTINET_ARCHITECTURE.md).

| Net | 분류 | 사용 MVP | V100 wall time | Master rms |
|---|---|---|---|---|
| Net 1 — Crystal Cathedral | 5-bus 병렬 | A C D F G H I | ~1.5 s | 0.232 |
| Net 2 — Recursive Organ | 3-pass 매크로 피드백 | A C F G | ~2 s | 0.058 |
| Net 3 — Decoding Chamber | 직선 9 단 | A C D E F G I | ~2 s | 0.177 |
| **Net Max — Cathedral Hive** | 8-bus + cross-feedback + 2-pass | A B C D E F G H I (9개 모두) | ~24 s | 0.260 |
| **Net Dynamic — Tempest** | 8-bus + 시간 가변 envelope + filter sweep + 3 impulse | A B C D E F G H I (9개 모두) | ~20 s | 0.30 평균 (0.05–0.60) |

### 10.1 Net Max 의 8 버스 구조
*   `α Foundation`: I → C(invalid_token) → 80Hz crossover +10dB
*   `β Core`: A → D(guitar) → E → G → A(drop) → F (6 단 RAVE 도메인 손상)
*   `γ Ghost`: F → G → C(shuffle) → F (freeze 중첩)
*   `δ Twin`: D(organ t=0.995) → A → C → E (오르간 측 모핑)
*   `ε Glitch`: C(rate=0.10) → I(fold) → G (강한 손상)
*   `ζ Drone`: H(prime) + H(fibonacci) → A (생성 베드)
*   `η Loop-B`: B 캡션 → TTA depth=3 (의미 표류, 현재 stub)
*   `θ XFB`: β 출력 → G(deep) → I(fold) (cross-bus 피드백)

Pass 1 + Pass 2 (refed seed 로 재실행) + S-curve crossfade 로 MASTER_FINAL 생성. 6개의 독립 음정 영역이 동시 점유 (sub 49–82 Hz / 207 / 337 / 432–444 / 439–442 / 660).

### 10.2 Net Dynamic 의 시간 가변 다이내믹
같은 8 버스를 한 번 렌더한 후 post-automation 단계에서:
*   버스별 진폭 envelope (구간별 선형 + 0.3 s 스무딩) — 60 초 작곡 arc 형성
*   3 impulse 이벤트: 15s freeze CLICK, 30s SILENCE DROP, 45s drone BURST
*   Master 저역통과 sweep: 250 Hz → 12 kHz → 500 Hz → 16 kHz → 6 kHz 시간 가변
*   초당 master RMS 범위 0.046 ~ 0.605 → **22 dB 다이내믹 레인지** (Net Max 의 5배)

---

## 11. Meta-Symphony — 매크로넷의 네트워크 (`scripts/meta_symphony.py`)

4 개 매크로넷의 출력을 3 분 stereo 타임라인 위에서 엮는 최상위 작곡. AudioArt 스택의 첫 stereo 출력. 자세한 설계: [`META_SYMPHONY_ARCHITECTURE.md`](META_SYMPHONY_ARCHITECTURE.md).

*   **Phase 1 스템 생성** (mono, 180 s 시드): Net 1 (raw) → Net 3 (RMS_match seed←Net 1), Net 2 (raw, 2-pass) → Net Dynamic (RMS_match seed←Net 2). 두 직렬 의존 chain 으로 같은 시드를 *이미 뉴럴 처리된* 형태로 다음 net 에 전달.
*   **Phase 2 인터위빙** (stereo): LFO crossfade `lfo_A=½+½sin(2π·t/60)` 와 `lfo_B=½+½cos(2π·t/45)`; 스테레오 pan `pan_A=0.7sin(2π·t/20)` + `pan_B=0.7cos(2π·t/25)`. LCM(60,45)=180 초로 곡 끝에서 위상 정렬, LCM(20,25)=100 초로 Lissajous.
*   **Phase 3 Foundation 강화**: 100 Hz 2 차 Butterworth LPF + 시드 +8 dB sub-boost 양 채널 재주입, tanh limiter drive=1.25, peak 0.95 정규화.

V100 단일 GPU wall time 약 3 분. 최종 `META_SYMPHONY_FINAL.wav` ~ 35 MB (180 s × stereo × 48 kHz).

---

## 12. 정적 데모 페이지 (`scripts/build_demo.py` → `demo.html`)

운영자 / 외부 청취자가 모든 단계 출력을 한 페이지에서 검증할 수 있도록 정적 HTML 데모를 빌드.

*   GitHub dark 톤 (#0d1117 배경, #7fffd4 accent), sticky top nav, responsive 카드 그리드.
*   각 트랙 카드: 제목 + 한국어 설명 + waveform PNG (320×80 px) + native `<audio controls preload="none">` + 메타데이터 (duration, SR, channels, RMS, peak, file size) + 다운로드 링크.
*   9 섹션 / 52 트랙: Seeds (4) · Net 1 (6) · Net 2 (4) · Net 3 (8) · Net Max p1 (9) · Net Max p2 + Final (5) · Net Dynamic (9) · Meta-Symphony (1) · Archive (6).
*   Waveform 썸네일은 `runs/_thumbs/<safe_id>.png` 에 캐시. wav가 변경된 트랙만 재생성.
*   실행: `python scripts/build_demo.py` → `demo.html` (32 KB).
*   재생: `python -m http.server 8765 --bind 127.0.0.1` 후 `http://localhost:8765/demo.html`.

Manifest 는 `build_demo.py` 안에 SEEDS / NET1 / NET2 / NET3 / NET_MAX_PASS1 / NET_MAX_PASS2_FINAL / NET_DYN / META / ARCHIVE 8 개 Python 리스트로 분리. 새 트랙 추가 시 해당 리스트에 한 줄 추가 후 재실행.

---

## 13. 향후 작업 — AudioLLM 통합

프로젝트 명칭 "AudioLLM-Art" 가 가리키듯, 현재까지는 *뉴럴 사운드 아트* 단계의 V1 구현. 다음 단계의 V2 는 다음을 포함한다:

1. **MVP-B 실제 백본 활성화** — `caption.backend=qwen2_audio` + `tta.backend=audioldm2` 로 전환. Qwen2-Audio-7B-Instruct (~14 GB) + AudioLDM2 (~4 GB) 다운로드. Net Max / Net Dynamic 의 η 버스가 의미적으로 살아 움직임.
2. **Semantic Governor** — Texture Governor 와 짝을 이루는 의미 측면의 가드. 청크 캡션이 시드 의미와 너무 멀어지면 자동 wet 감쇠.
3. **AudioLLM 조건부 매크로넷** — `net_semantic`, `net_llm_chain`, `net_prompt_morph` 등 캡션 텍스트가 다른 net 의 파라미터를 시간 가변으로 결정하는 새 토폴로지.
4. **Meta-Symphony v2 — 5-stem 인터위빙** — 현재의 4 stem 에 `stem_LLM` 추가, 멀티 모달 텍스트-사운드 cross-modal loop.
5. **AudioLLM 자체 손상** — 캡션 모델의 텍스트 임베딩 공간에 직접 노이즈 주입, 의미 표류의 "결" 자체를 변형. *AudioLLM 도메인의 perturbation*.
6. **재현성 패키지 자동 생성** — 각 마스터피스 wav 와 함께 git hash + config + 모델 SHA + LUFS 측정 결과를 sidecar JSON 으로 묶는 빌더.

V1 의 최종 결과 (`runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav` 등) 는 *Pre-AudioLLM* 라벨로 보존된다.
