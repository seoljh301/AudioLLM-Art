# MVP-D — Checkpoint Morphing (옵션: MVP-A perturb 체인)

## 가설
체크포인트는 "세계에 대한 모델의 얼어붙은 청각적 해석"이다. 서로 다른 corpus 로 학습된 두 체크포인트 (예: 타악기 vs 음성) 를 보간하면, 두 악기 어느 쪽도 아니면서 동시에 둘 다이기도 한 *하이브리드 청취 기계* 가 만들어진다.

## 모델 의존성
- 동일 아키텍처의 RAVE `.pt` 또는 `.ts` 체크포인트 ≥ 2개.
- (옵션) DAC 체크포인트 — codec 레벨 보간용.

## 예상 음상

| 모드 | 결과 |
|---|---|
| `linear`, t ∈ [0,1] | 두 음색 세계 간의 부드러운 crossfade |
| `slerp` | latent 기하 위의 호 — midpoint 가 종종 더 "살아 있는" 음색 |
| `random_walk` | 확률적 morph — 표류하는 불가능 악기 변주 |

## Sweep 가능한 파라미터

- `morph.t ∈ [0, 1]` (실시간 변경 가능)
- `morph.mode ∈ {linear, slerp, random_walk}`
- `morph.walk_step ∈ [0.01, 0.2]`
- `rebasin: true/false`, `rebasin_mode ∈ {partial, full}`

## 실행

### 0. 체크포인트
동일 아키텍처의 RAVE TorchScript export ≥ 2개 (같은 `latent_dim`, `sample_rate`, `hop`).

검증용 stub:
```bash
python scripts/make_stub_rave.py --out checkpoints/rave/stub_a.ts --seed 0
python scripts/make_stub_rave.py --out checkpoints/rave/stub_b.ts --seed 1
```

실제 모핑은 동일 아키텍처 export 2개를 다운로드 (예: 둘 다 `b2048_r48000_z16`) — [`acids-ircam/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models) 추천. `config.yaml` 의 `checkpoints.paths` 가 가리키게 설정.

> ⚠️ `.ts` 로딩은 TorchScript 코드 실행 — 신뢰 가능한 출처만.

### A+D 체인
MVP-A 스타일 latent perturb 를 모핑된 모델 위에 적층하려면 `config.yaml` 에 `perturb:` 블록 추가:

```yaml
morph: {mode: linear, t: 0.005}
perturb: {noise_scale: 0.5, dim_dropout: 0.3}
```

`results.json` 의 `ad_chain_sweep` 와 `dropout_intensity_sweep_bilateral` 참고. 양측 7×6 dropout sweep 결과:

- Guitar 쪽: d ≈ 0.3–0.5 까지 단조 증폭 (1.9× base rms), d=0.7 에서 collapse.
- Organ 쪽: d ≈ 0.1–0.2 에서 작은 증폭, d=0.3 에서 뚜렷한 dip, d=0.5 부분 회복, d=0.7 collapse.

서로 다른 latent 기하: guitar 는 많은 inhibitory 차원이 있어 (dropout 늘수록 나머지가 더 크게 발화), organ 은 d ≈ 0.3 근처에 모인 소수의 결정적 harmonic 차원만 가지고 있다. perturbation 으로 중간 t 의 silent plateau 를 구할 수 없다.

### 1. 오프라인 렌더 (단일 morph 스냅샷)
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

### 2. OSC 서버 (실시간 remorph)
```bash
bash run.sh --mode serve
```

## OSC 인터페이스

| 주소 | 인자 | 동작 |
|---|---|---|
| `/mvp_d/t` | float | 모핑 계수 — state_dict 재합성 트리거 |
| `/mvp_d/mode` | string | linear / slerp / random_walk 전환 |
| `/mvp_d/render` | string in, string out | 렌더 + `/mvp_d/done <out>` 응답 |

## 하드웨어 노트
- V100 GPU 지원: 루트 `environment.yaml` 의 `torch==2.4.1 + cu121` 핀.
- Remorph 자체는 빠르다 (state_dict 크기 ≈ 모델 크기 / 키 당 텐서 연산 1번); 총 시간은 렌더가 지배.

## 알려진 실패 모드

독립 학습된 두 RAVE 체크포인트 (테스트 사례: guitar + organ) 의 naive 선형 보간은 **모든 중간 t 에서 거의 무음으로 collapse** 한다 (endpoint 에서 rms ≈ 0.4 vs 중간 t 에서 ≈ 0.0005). `t=0` 과 `t=1` 은 깨끗하게 원본 모델을 만드는데, 그 사이는 모델을 텅 비워버린다.

이는 뉴럴 네트워크 mode connectivity 의 알려진 성질 — 독립 학습된 네트워크는 weight 공간에서 동일 low-loss 곡선 위에 놓이지 않는다.

### 회피 방법

1. **프리셋으로 활용** — "Hollowed-out RAVE" 자체가 그 자체로 쓸 만한 사운드 아트 텍스처.
2. **공유 init 으로부터 두 RAVE 학습** — 깨끗하게 모핑되어야 함.
3. **Git Re-Basin / 순열 정렬** — interp 전에 한 모델의 hidden unit 을 다른 모델에 맞춰 재순열. 두 구현 시도:
   - **Partial** (`src/modules/mvp_d/re_basin.py`, `rebasin_mode="partial"` — 기본): 각 paired resblock 의 inner 채널 정렬 (21 블록, grouped-conv 인지). `.1.weight` 출력 + `.3.weight` 입력에 LAP. **Collapse 방지 안 됨.**
   - **Full encoder** (`src/modules/mvp_d/re_basin_full.py`, `rebasin_mode="full"`): chain class `enc_48 / 96 / 192 / 384` + inner class + grouped conv 상류 채널의 tied-group 의미론까지. Ainsworth 2023 Algorithm 3 의 5-iter weight matching. Random-perm 검증으로 함수 보존 확인. 하지만 **모핑은 여전히 collapse** — encoder 정렬만으로는 부족하다. latent 출력은 (하류 consumer 일관성 유지를 위해) 순열되지 않으며, COLLAPSE 는 하류 (decoder) 에 있다. 독립 학습된 두 악기의 decoder 가중치가 naive 블렌드되면 → silence. 완전한 fix 는 decoder + gimbal + prior_net + latent_pca chain 모두를 정렬해야 한다. `results.json` 의 `rebasin_full_attempt` 비교 데이터 참고.
4. **Endpoint 근처 머무르기** — fine sweep 으로 확인됨. `t ∈ [0, 0.02]` 와 `t ∈ [0.98, 1.0]` 이 cliff 직전의 깨끗한 "악기 페이드" 음악적 영역. silent plateau 는 `t ∈ [0.10, 0.90]`. `results.json` 의 `cliff_sweep_guitar_organ` 와 `runs/mvp_d/cliff/` 의 렌더 19개 참고.

Stub 체크포인트 쌍 (`stub_a.ts` + `stub_b.ts`, 둘 다 random init) 은 단조 linear morph 가 **잘 된다** — 동일 아키텍처를 공유하면서 random 으로 호환되기 때문. 실제 학습된 네트워크는 서로 다른 basin 을 차지한다.

## 향후 작업

- **공유 init RAVE 학습 페어** — 같은 시드로 시작해 다른 corpus 학습. 직접 검증.
- **Decoder + gimbal + prior_net + latent_pca 까지 확장한 full Re-Basin** — 현재 encoder 만 정렬. RAVE 코드 읽고 permutation symmetry 그래프 완성 필요.
- **AudioLLM 조건부 morph_t** — 캡션 텍스트가 시간 가변 t 를 결정하는 prompt-conditioned morphing.
- **시간 가변 t LFO 자동화** — 곡 진행에 따라 endpoint cliff 영역을 LFO 로 오가는 자동 변조.
