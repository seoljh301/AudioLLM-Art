# MVP-Q — Phase Scramble

**위치:** matrix `audio → audio · STFT`.

## 가설
magnitude 가 timbre/pitch 를 거의 다 들고 있고, phase 는 transient/공간감을 결정한다. magnitude 유지 + phase 만 변조 → 같은 음색의 시간축 박살난 버전.

## 모드
| mode | 효과 |
|---|---|
| `random_uniform` | phase ~ U(-π, π) — 가장 파괴적, 트랜지언트 제거 |
| `frame_swap` | 시간 프레임 순서 셔플 (phase 만) |
| `bin_swap` | 주파수 bin 셔플 (프레임마다) |
| `rotate` | 주파수 의존 일정 회전 — 주파수 도메인 지연 |
| `ou` | OU 과정 — 부드러운 phase 변동 |

## 사용 예
```bash
bash experiments/mvp_q_phase_scramble/run.sh \
  --target runs/multinet/sine_440_10s.wav \
  --out runs/mvp_q/scramble_rand_10s.wav
```
