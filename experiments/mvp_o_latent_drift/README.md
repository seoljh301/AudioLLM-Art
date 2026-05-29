# MVP-O — Latent Drift Generator

**위치:** matrix `∅ → audio · latent`. MVP-A 의 입력 없는 쌍대.

## 가설
입력 신호 없이 RAVE 잠재 공간에 직접 분포를 부여하면, 디코더가 단독으로 환각 소리를 만들어낸다. RAVE 가 "리얼한 입력" 없이 어떤 음향 사전 분포를 갖는지 직접 들음.

## 모드
| mode | z(t) 분포 |
|---|---|
| `white` | iid N(0,σ²) |
| `smoothed` | IIR 저주파 white → Brownian 산책 |
| `ou` | Ornstein-Uhlenbeck mean reversion |
| `sinusoid` | dim 별 저주파 사인파 합 |

## 사용 예
```bash
bash experiments/mvp_o_latent_drift/run.sh --out runs/mvp_o/drift_001.wav
```
