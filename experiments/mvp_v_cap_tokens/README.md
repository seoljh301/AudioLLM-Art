# MVP-V — Caption-Conditioned Tokens

**위치:** matrix `audio → text → audio · tokens`.

## 가설
audio → caption → caption hash → EnCodec 코드북 위에서 결정론적 token walk → decode. caption 이 코드북 궤적의 prior 로 작동. 청크별로 caption 갱신 = 시간 가변 token grammar.

## walk_mode
| mode | 의미 |
|---|---|
| `prime` | 소수 시퀀스로 +/- step (deterministic) |
| `fibonacci` | 피보나치 차분 step |
| `random_walk` | uniform [-strength, +strength] step |

## 사용 예
```bash
bash experiments/mvp_v_cap_tokens/run.sh \
  --target runs/multinet/sine_440_10s.wav \
  --out runs/mvp_v/cap_tokens_prime.wav
```
