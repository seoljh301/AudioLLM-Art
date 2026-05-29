# MVP-R — Token Musaicing

**위치:** matrix `corpus × target → audio · tokens`. MVP-M (latent) 의 토큰 쌍대.

## 가설
EnCodec / DAC 의 이산 토큰 도메인에서 직접 musaicing 하면, 연속 latent (M) 보다 더 강한 양자화 잡음 + concatenative 미학이 결합된다. 토큰 단위 voting 으로 grain 경계가 자연스럽게 평균화.

## 핵심
- corpus 인코딩 → (n_q, T) → 윈도우 (n_q, G) grain bank
- target 인코딩 → 윈도우별 Hamming/weighted-Hamming/first-q 거리
- softmax(-dist/τ) 로 grain 선택
- 토큰 공간에서 stride concat + 셀 단위 majority vote
- 한 번에 decode

## 거리 모드
| dist | 의미 |
|---|---|
| `hamming` | 전 quantizer 동등 가중 |
| `weighted_hamming` | q_weights 로 quantizer 별 가중 (RVQ 의 상위 quantizer 더 중시) |
| `first_q` | 첫 quantizer 만 사용 — coarse match |

## 사용 예
```bash
bash experiments/mvp_r_token_musaicing/run.sh \
  --corpus runs/multinet/net1/bus_M_core.wav runs/multinet/net1/bus_H_shimmer.wav \
  --target runs/multinet/sine_440_10s.wav \
  --out runs/mvp_r/tok_001.wav
```
