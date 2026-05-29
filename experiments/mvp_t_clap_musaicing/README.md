# MVP-T — CLAP / Text-Driven Musaicing

**위치:** matrix `corpus × text → audio`.

## 가설
M/N 은 audio target → grain match. T 는 **text** target → grain match. 동일 corpus 라도 "deep cavernous drone" 과 "bright shimmer" 의 prompt 가 완전히 다른 시퀀스 선택.

## 백엔드
- `use_real: false` → procedural caption→hash embed (deterministic)
- `use_real: true` → LAION-CLAP. 실패 시 fallback.

## 사용 예
```bash
bash experiments/mvp_t_clap_musaicing/run.sh \
  --corpus runs/multinet/net1/bus_M_core.wav runs/multinet/net1/bus_H_shimmer.wav \
  --prompts "bright shimmer with metallic resonance" "deep cavernous drone" \
  --out runs/mvp_t/shimmer_then_drone.wav
```
