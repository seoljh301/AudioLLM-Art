# MVP-W — Caption → TTA Recursive Loop

**위치:** matrix `audio → text → audio · STFT`.

## 가설
caption(audio) → TTA(caption) → 다시 caption → ... 반복하면 결국 caption banker 의 attractor 로 흡수된다. iteration 별 drift 를 들음.

## 사용 예
```bash
bash experiments/mvp_w_cap_loop/run.sh \
  --target runs/multinet/sine_440_10s.wav \
  --out-dir runs/mvp_w/loop_001
```

iter_01.wav, iter_02.wav, ..., final.wav 저장.
