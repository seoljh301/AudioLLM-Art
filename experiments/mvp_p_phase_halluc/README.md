# MVP-P — Phase Hallucinator

**위치:** matrix `∅ → audio · STFT`.

## 가설
임의 magnitude spectrogram + Griffin-Lim 으로 phase 를 환각해도 청취 가능한 사운드가 나온다. "spectrum 만 의도하고 phase 는 모델/알고리즘이 채워넣게 하기" 의 단순한 형태.

## 모드
| mode | 핵심 |
|---|---|
| `white` | uniform 노이즈 mag |
| `pink` | 1/√f 감쇠 |
| `shaped_smooth` | 시간/주파수 IIR 스무딩된 노이즈 |
| `chord` | 화음 기본 + 고조파 + 시간 변조 |
| `comb` | 빗 모양 (등간격 피크) |

## 사용 예
```bash
bash experiments/mvp_p_phase_halluc/run.sh --out runs/mvp_p/halluc_chord_10s.wav
```
