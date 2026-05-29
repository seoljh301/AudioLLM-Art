# MVP-U — Caption-Steered Latent

**위치:** matrix `audio → text → audio · latent`.

## 가설
audio → caption → text embedding 을 RAVE 잠재 공간에 선형 사영해서 latent stream 에 bias 로 더한다. 청크 단위로 재캡션 → 시간 가변 bias. 텍스트-만든 잠재 경로.

## 사용 예
```bash
bash experiments/mvp_u_cap_latent/run.sh \
  --target runs/multinet/net1/bus_M_core.wav \
  --out runs/mvp_u/cap_steer_001.wav
```
