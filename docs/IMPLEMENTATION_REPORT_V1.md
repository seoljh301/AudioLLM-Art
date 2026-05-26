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

## 7. 결과물 위치 및 최종본
*   **최종 작품**: `runs/masterpiece/final_symphony_bass_heavy.wav`
*   **안정성**: 180초 전 구간 결함 없음.
*   **질감**: 8개 레이어의 유기적 중첩 및 LFO 오토메이션.
*   **에너지**: -12 LUFS의 높은 음압과 +8dB의 서브 베이스 타격감.
