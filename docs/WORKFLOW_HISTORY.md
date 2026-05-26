# AudioArt Workflow & File History (2026-05-22)

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

## 4. 순수 뉴럴 체인 및 MVP-E 탄생
*   **폴더**: `runs/feedback_final5/`
*   **공정 (A-D-D-A-A-D-A)**: MVP-B(텍스트)를 제거한 순수 잠재 공간 표류 실험.
*   **MVP-E (Neural Granular)**: 40초간의 Latent 기억을 현재로 투사하는 '시간 지체' 질감 구현.

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
