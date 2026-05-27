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
1시간(약 68분) 분량의 대용량 오디오를 여러 겹으로 쌓아 올릴 때 발생하는 메모리 초과(OOM) 현상과 파이썬 프로세스 강제 종료(SIGKILL) 문제를 해결한 기술적 명세입니다.

### 7.1 SoX (Sound eXchange) 커맨드라인 프로세싱
*   **문제**: `librosa`를 이용한 실시간 리샘플링과 메모리 상의 거대 행렬(Float32, 68분, 8채널) 연산이 시스템 RAM을 100% 점유하여 강제 종료됨.
*   **해결**: 파이썬의 오디오 연산 의존도를 낮추고, C 기반의 최적화된 오디오 프로세싱 도구인 **SoX**를 도입.
*   **구현 (`scripts/run_hfo_master_sox.sh`, `scripts/run_f0_octave_sox_master.sh`)**:
    *   디스크 스트리밍 방식의 필터링(`sinc`), 피치 시프팅(`pitch`), 멀티트랙 믹싱(`-m`)을 백그라운드 쉘 스크립트로 분리하여 실행.
    *   **결과**: 메모리 누수 없이 68분짜리 오디오 8개를 10여 분 만에 안정적으로 합성에 성공.

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

---

## 9. 결과물 위치 및 최종본
*   **최종 작품**: `runs/masterpiece/final_symphony_bass_heavy.wav`
*   **안정성**: 180초 전 구간 결함 없음.
*   **질감**: 8개 레이어의 유기적 중첩 및 LFO 오토메이션.
*   **에너지**: -12 LUFS의 높은 음압과 +8dB의 서브 베이스 타격감.

