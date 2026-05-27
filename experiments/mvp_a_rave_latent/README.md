# MVP-A — RAVE Latent Perturbation

## 가설
RAVE 의 latent `z` 를 perturb (가우시안 노이즈, dim dropout, dim shuffle, additive bias) 시키면 *악기 정체성을 유지한 채* 미적으로 의미 있는 음색 표류가 발생한다. perturb 강도가 임계치를 넘는 순간 모델이 환각을 시작한다.

## 모델 의존성
- 사전 학습된 traced RAVE `.ts` 모델을 `checkpoints/rave/<name>.ts` 에 둘 것.
- 공식 RAVE model zoo 에서 다운로드하거나 IRCAM RAVE repo 로 직접 학습.

## 예상 음상

| `noise_scale` | 결과 |
|---|---|
| ~0.05 | 미세한 음색 떨림 |
| ~0.3 | 일그러진 텍스처, partial 이동 |
| > 0.8 | 의사-악기 옹알이, 모델이 자기 정체성을 잊기 시작 |

## Sweep 가능한 파라미터

- `noise_scale ∈ [0, 1]`
- `dim_dropout ∈ [0, 0.5]`
- `dim_shuffle ∈ {false, true}`
- `noise_mode ∈ {white, smoothed}` — smoothed 는 1차 IIR Brownian 드리프트 (`noise_smooth=0.98`)
- `freeze_mask` — 어떤 latent 차원을 고정할지

## 실행

### 0. 모델 준비
검증만을 위한 stub:
```bash
python scripts/make_stub_rave.py --out checkpoints/rave/stub.ts
```

실제 음상을 원하면 사전 학습된 RAVE TorchScript export 를 다운로드 — 예를 들어 [`acids-ircam/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models) (`voice_jvs_b2048_r44100_z16.ts`, `guitar_iil_b2048_r48000_z16.ts` 등). `config.yaml` 의 `model.path` 가 다운로드한 파일을 가리키게 한다.

> ⚠️ `.ts` 파일을 로드하면 임의 TorchScript 코드가 실행된다. 신뢰 가능한 출처에서만 가져올 것.

### 1. 오프라인 렌더
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

### 2. OSC 서버 (Max 에서 `/mvp_a/render <in> <out>` 전송)
```bash
bash run.sh --mode serve
```

## OSC 인터페이스

| 주소 | 인자 | 동작 |
|---|---|---|
| `/mvp_a/noise` | float | noise_scale 설정 |
| `/mvp_a/dropout` | float | dim_dropout 설정 |
| `/mvp_a/shuffle` | int 0/1 | dim_shuffle 토글 |
| `/mvp_a/render` | string in, string out | 렌더 트리거, `/mvp_a/done <out>` 응답 |

## 하드웨어 노트
- V100 GPU 지원: 루트 `environment.yaml` 의 `torch==2.4.1 + cu121` 핀에 의존. cu130 wheel 은 SM ≥ 7.5 가 필요해 V100 에서 silently fail.
- V100 위 실제 RAVE: 5 초 오디오를 1.5–2 초 wall time 에 렌더 (≈ 3× 실시간).

## 안전장치 (자동 적용)
모든 렌더는 `src/core/mix.py` 의 dry/wet + soft limiter + (옵션) 80 Hz crossover anchor 그리고 `src/core/texture_governor.py` 의 청크별 자동 wet 감쇠를 통과한다. NaN / 폭주 / silence 모두 출력 도달 전에 차단된다.

## 향후 작업

- **Latent dim mapping 자동화** — RAVE 모델별로 latent 의 어느 차원이 어떤 perceptual axis (pitch / timbre / dynamics) 와 상관되는지 매핑.
- **AudioLLM 조건부 perturbation** — 향후 MVP-B 의 실제 캡션 모델이 활성화되면, 캡션 텍스트의 임베딩을 noise 의 방향성에 사상 → "metallic" 형용사가 들어오면 특정 latent 차원만 강조하는 의미적 perturb.
- **MVP-D 와 결합 (A+D 체인)** — 이미 `experiments/mvp_d_ckpt_morph/` 에서 ad_chain 으로 실험됨. 추가로 morph t 가 시간 가변일 때 perturb 도 동기화하는 LFO 체인 검토.
