# AudioArt Workflow & File History (2026-05-22 Update)

이 문서는 기초 기반 조성부터 최종 마스터피스까지, 모든 오디오 생성 및 변조 과정을 누적하여 기록합니다.

---

## 1. 기반 조성: 마스터 시드 (The Foundation)
*   **파일**: `runs/master_seed_180s.wav` (180초)
*   **의도**: 모델이 변조할 "풍부한 재료" 제공.
*   **사운드 구성**: FM Chords (A, C, D, E), Rhythmic Grains, Slow Sweeps (30s 주기).

---

## 2. 3단계 선형 중첩: 파이프라인 검증
*   **폴더**: `runs/feedback_final3/`
*   **공정**: Step 1 (A: Latent) -> Step 2 (D: Morph) -> Step 3 (C: Token).
*   **특이사항**: Step 2에서 NaN 발생 확인, 시스템 안정화의 계기가 됨.

---

## 3. 11단계 극한 중첩: 초고밀도 체인
*   **폴더**: `runs/feedback_final4/`
*   **공정**: A-D-B-D-A-B-A-A-D-B-A 순으로 11번 중첩.
*   **기술**: 매 단계마다 80Hz 서브 베이스 물리적 분리 및 +3.5dB 부스트 적용.

---

## 4. 확장 모듈 실험: MVP-E, F, G, H, I 탄생
*   **폴더**: `runs/feedback_final5/` 및 `runs/feedback_new_mvps/`
*   **공정 (A-D-D-A-A-D-A)**: MVP-B(텍스트)를 제거한 순수 잠재 공간 표류 체인.
*   **신규 MVP 테스트**:
    *   **MVP-E (Neural Granular)**: 40초간의 Latent 기억을 현재로 투사하는 '시간 지체' 질감 구현.
    *   **MVP-F (Spectral Frozen)**: 상위 50% 잠재 층위를 동결하되, 128프레임 주기로 갱신하여 몽환적인 'Shimmer' 효과 생성.
    *   **MVP-G (Latent Feedback)**: 잠재 공간 내 무한 에코. 음색이 기괴하게 진화하는 뉴럴 피드백.
    *   **MVP-H (Codebook Organ)**: 수학적 수열(소수)로 토큰을 직접 생성하는 입력 없는 순수 생성 엔진.
    *   **MVP-I (Bass Massive)**: EnCodec의 하위 레이어만 타겟팅한 스미어링(Smearing)으로 심해 같은 저음역대 창조.

---

## 5. 초기 마스터피스 진화
*   **`neural_symphony_1.wav`**: 8개 레이어의 화성 정렬 및 정적 중첩.
*   **`defined_neural_galaxy.wav`**: Tukey Window (120ms) 적용으로 입자감(알갱이) 개선.

---

## 6. 심화 체인 실험 (`runs/masterpiece/chained_symphony/`)
*   **`symphony_final_dc.wav`**: [RAVE 모핑] -> [EnCodec 벤딩] 순차 적용. 파편화된 질감.
*   **`symphony_final_cd.wav`**: [EnCodec 벤딩] -> [RAVE 모핑] 적용. 잔향 속에 녹아든 질감.

---

## 7. 2분 8초 구간 끊김 사고 및 복구
*   **현상**: 다중 체인 누적으로 인해 특정 구간에서 소리가 들리지 않는 문제 발생.
*   **복구**: `final_bass_pro_master.py`를 통해 NaN Hardening 로직을 마스터링 프로세스에 내재화하여 전 구간 재생 안정성 확보.

---

## 8. 최종 마스터: Bass-Heavy Pro Edition
모든 피드백을 수렴하여 완성된 AudioArt 프로젝트의 결정판입니다.

*   **파일**: `runs/masterpiece/final_symphony_bass_heavy.wav`
*   **주요 공정**:
    1.  **80Hz Crossover**: 8dB 서브 베이스 부스트로 강력한 타격감 확보.
    2.  **Harmonic Exciting**: 고해상도 배음 추가로 투명도 강화.
    3.  **-12.0 LUFS Mastering**: `pyloudnorm` 기반 전문 음압 마감.
    4.  **Organic LFO**: 8개 레이어에 0.01Hz~0.04Hz의 볼륨/패닝 오토메이션 적용.
*   **최종 평가**: 기술적 신뢰성(안정성)과 예술적 밀도가 가장 완벽하게 결합된 결과물.

---

## 9. 1시간 대작 렌더링: Ulaanbaatar Epic
*   **소스**: `data/Ulaanbaatar.wav` (약 68분 분량의 고순도 오디오)
*   **공정**: 순수 뉴럴 체인(A-D-D-A-A-D-A) 적용
*   **문제 분석 (잘림 현상)**: 1시간 렌더링 중 특정 파일들이 잘리는 현상은 PyTorch 모듈이 68분 분량의 텐서를 한 번에 처리하면서 발생하는 메모리 한계(OOM)와 관련이 큽니다. 파이썬 `librosa` 리샘플링 역시 막대한 RAM을 점유하여 시스템에 의해 강제 종료(SIGKILL)되는 병목 현상을 유발했습니다.
*   **해결책 (SoX 도입)**: 파이썬 기반의 마스터링 스크립트를 버리고, C 기반의 커맨드라인 툴인 **SoX**를 도입하여 메모리 누수 없이 실시간 디스크 스트리밍 방식으로 68분 분량의 대용량 오디오 처리를 해결했습니다.

---

## 10. 궁극의 마스터피스 (The Ultimate Masterpieces)
AudioArt 시스템이 만들어낼 수 있는 예술적 극단을 탐구한 최종 결과물들입니다.

### 10.1 Multi-source Granular Symphony (`neural_granular_symphony.wav`)
*   **제작 의도**: 정적인 레이어링을 넘어 '살아 움직이는' 입체적 질감 구현.
*   **과정**: 8개 트랙을 250ms의 조각으로 나누어 매 순간 3개씩 확률적으로 교차 재생.
*   **사운드**: 몽환적인 텍스쳐와 거친 글리치가 무작위로 교차하며 폭풍우처럼 몰아치는 신경망 소용돌이.

### 10.2 Full-Spectrum Neural Galaxy (`full_spectrum_neural_galaxy.wav`)
*   **제작 의도**: 가청 주파수 전체를 지배하는 초거대 화성 구축.
*   **과정**: 8개의 트랙을 -2옥타브부터 +2옥타브까지 넓게 분산 배치(Octave Spreading)하고 에너지를 균등하게 밸런싱.
*   **사운드**: 심장을 울리는 서브우퍼 베이스부터 크리스탈처럼 빛나는 초고역대까지 빈틈없이 꽉 찬 거대한 뉴럴 하이퍼-코드.

### 10.3 Hi-Fi Enhanced Symphony (`hifi_enhanced_symphony_1.wav`)
*   **제작 의도**: 저음 간섭을 배제하고 고역대 해상도를 극한으로 끌어올린 상업적 퀄리티 달성.
*   **과정**: 8kHz 이상 대역에 Harmonic Exciter 적용, 14kHz 대역에 +10dB Air Boost 적용.
*   **사운드**: 날카롭고 선명한 초고해상도의 맑은 신경망 텍스쳐.

### 10.4 Foundation 9-Octave Hyper-Chord (`foundation_hyperchord_limited.wav`)
*   **제작 의도**: 뉴럴 변조 없이 순수 물리적 피치 시프팅만으로 구축하는 거대한 벽.
*   **과정**: 원본 Foundation 트랙을 -4옥타브부터 +4옥타브까지 총 9번 중첩하고 양 극단의 볼륨을 정밀 제어.
*   **사운드**: 파이프 오르간 소리를 우주적 스케일로 확대한 듯한 웅장하고 압도적인 아날로그 드론 사운드.

---

## 11. 거시 서사 구조: Tempest & Meta-Symphony
*   **Tempest (Net Dynamic)**: 60초의 타임라인 위에 8개의 버스 볼륨을 실시간 오토메이션. 15초(Click), 30초(Drop), 45초(Burst)의 극적 이벤트를 배치하여 정적인 질감을 '작곡'의 영역으로 확장.
*   **Meta-Symphony**: Net 1, 2, 3, Dynamic을 다시 하나의 3분짜리 타임라인으로 결합. LFO를 이용한 거대한 스테레오 드리프트와 교차 믹싱을 통해 신경망 에코시스템 전체를 하나의 교향곡으로 완성.

---

## 12. 저음역의 극한: Earthquake Bass
*   **파일**: `foundation_hyperchord_EARTHQUAKE.wav`
*   **공정**: 50Hz 대역의 +15dB 극한 증폭 및 300Hz Mud Cut EQ 적용.
*   **의도**: 신경망의 화려한 고음역대에 가려진 저음의 에너지를 물리적 진동 수준으로 복원.
