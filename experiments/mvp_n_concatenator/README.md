# MVP-N — The Concatenator

**Paper:** Tralie & Cantil, *The Concatenator: A Bayesian Approach To Real Time Concatenative Musaicing*, 2024 (arXiv:2411.04366)

## 가설

Particle filter + KL-divergence NMF observation model 로 target 의 spectrogram 을 코퍼스 윈도우 조합으로 재합성한다. 시간적 연속성 (`pd`) 과 다양성 (`τ`) 을 사용자가 직접 제어. 복잡도 `O(LPMpT)` — 코퍼스 크기 `N` 에 무관.

## 파이프라인

```
corpus ──STFT(mag,phase)──▶ W (M, N)
                           Φ (M, N)
target ──STFT(mag)────────▶ V (M, T)

for t in 1..T:
  particles ← transition(pd)           # 각 인덱스 prob pd 로 +1, else 무작위 점프
  for each particle k:
      h_k ← KL-NMF(v_t, W[:, particles_k], L iters)
      d_k ← KL(v_t || W[:, particles_k] · h_k)
  weights ∝ weights * exp(-d / τ)
  if ESS < threshold * P: resample
  V̂_t ← Σ_k w_k · W[:, particles_k] · h_k
  Φ̂_t ← phase from dominant particle's strongest activation

ISTFT(V̂, Φ̂) → output
```

## 주요 파라미터

| 이름 | 의미 | 권장 범위 |
|---|---|---|
| `P` | particle 수 | 100–10000 |
| `p` | particle 당 활성 코퍼스 윈도우 수 | 5 |
| `pd` | 시간적 연속성 확률 | 0.9–0.99 |
| `tau` | softmax 온도; 작을수록 likelihood 가 뾰족함 | 1–50 |
| `L` | KL-NMF 반복 횟수 | 10 |
| `ess_threshold` | 리샘플 트리거 (P 비율) | 0.5 |
| `l2_reg` | 조용한 구간용 L2 (Eq.10) | 0.0–0.1 |

## 사용 예

```bash
bash experiments/mvp_n_concatenator/run.sh \
  --corpus data/corpus/*.wav \
  --target runs/sine_30s.wav \
  --out runs/mvp_n/concat_001.wav
```

## V2 (AudioLLM)

- 코퍼스 윈도우에 텍스트 임베딩(예: CLAP) 부착 → 텍스트 프롬프트로 prior 가중
- AudioLLM 의 시맨틱 토큰 시퀀스를 transition prior 로 사용
- 실시간 OSC 트리거: `/mvp_n/tau`, `/mvp_n/pd`
