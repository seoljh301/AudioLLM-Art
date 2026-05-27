# feedback_1.md

# AudioArt Feedback 1 — 노이즈 수렴 방지와 Texture Guard 구현안

## 0. 문서 목적

이 문서는 AudioArt repo의 현재 MVP 구조를 기준으로, **의도적 모델 오용 / token bending / latent perturbation / checkpoint collapse**가 단순한 노이즈나 silence로 수렴하지 않고, **음악적으로 쓸 수 있는 기묘한 텍스쳐**로 남도록 하기 위한 피드백과 구체적 코드 구현안을 정리한다.

핵심 목표는 다음과 같다.

> 모델을 더 세게 망가뜨리는 것이 아니라,  
> **모델이 겨우 오해할 수 있는 정도로만 괴롭히고, 원본 구조를 붙잡은 채 표면을 변형한다.**

즉, 이 문서의 방향은 **destruction**이 아니라 **anchored corruption**이다.

---

## 1. 현재 프로젝트 상태 요약

AudioArt repo는 Audio Foundation Model을 정상적인 음악 생성기로 쓰는 것이 아니라, 다음과 같은 실패 양상을 사운드아트 재료로 사용하는 프로젝트다.

- RAVE latent perturbation
- Audio ↔ caption ↔ text-to-audio recursive loop
- EnCodec / DAC token bending
- Checkpoint morphing / checkpoint collapse
- Max/MSP ↔ Python backend OSC 제어
- V100 환경에서 돌아가는 실험용 MVP 구조

현재 MVP는 크게 네 개다.

| MVP | 핵심 조작 | 현재 의미 |
|---|---|---|
| MVP-A | RAVE latent perturbation | latent damage processor |
| MVP-B | audio → caption → TTA recursive loop | recursive mishearing machine |
| MVP-C | EnCodec/DAC token bending | neural codec databender / codebook organ |
| MVP-D | RAVE checkpoint interpolation | checkpoint collapse synthesizer |

각 MVP는 기본적으로 **모델을 망가뜨리는 방향**으로는 잘 설계되어 있다.  
하지만 현재 구조는 아직 다음 장치가 약하다.

- 원본 구조 anchor
- damage amount 자동 제어
- upper/lower representation 선택적 손상
- local continuity 보존
- feedback loop의 entropy governor
- dry/wet 기반 안전장치
- collapse/silence/noise 감지

따라서 지금 필요한 것은 새 모델이 아니라, **망가짐을 붙잡는 레이어**다.

---

## 2. 핵심 문제: 왜 노이즈로 수렴하는가?

Audio model을 일부러 깨부술 때 노이즈로 수렴하는 이유는 단순하다.

### 2.1 모든 층위가 동시에 무너지기 때문

소리에는 여러 층위가 있다.

- onset / rhythm
- pitch contour
- low-frequency body
- timbre
- high-frequency residual
- reverb / space
- semantic identity
- local continuity

이 중 전부를 동시에 망가뜨리면, 귀가 붙잡을 구조가 사라진다. 그러면 결과는 대체로 다음 셋 중 하나가 된다.

```text
1. white-noise-like harsh texture
2. near-silence collapse
3. boring static drone
```

재밌는 텍스쳐는 완전한 무작위가 아니다.  
항상 최소 하나 이상의 구조적 anchor가 남아 있다.

### 2.2 decoder manifold 밖으로 너무 멀리 나가기 때문

Neural codec이나 RAVE decoder는 학습된 latent/token manifold 근처에서는 흥미로운 artifact를 만든다.  
하지만 manifold 밖으로 너무 멀리 나가면, decoder가 더 이상 “그럴듯한 소리”를 복원하지 못하고 다음처럼 된다.

- clipped artifact
- broadband noise
- silence
- unnatural impulse
- meaningless codec trash

따라서 목표는 다음이 아니다.

```text
random token
random latent
random checkpoint
random feedback
```

목표는 다음이다.

```text
original token near-manifold corruption
smoothed latent drift
upper residual quantizer corruption
dry/wet anchored damage
metric-governed feedback
```

---

## 3. 전체 설계 원칙

## 3.1 하나는 반드시 보존한다

각 실험마다 최소 하나의 구조는 보존한다.

| 보존할 것 | 망가뜨릴 것 | 예상 결과 |
|---|---|---|
| rhythm / onset | timbre | glitch groove |
| pitch contour | high-frequency detail | ghost instrument |
| lower RVQ | upper RVQ | 구조 유지 + 표면 손상 |
| dry input | damaged model output | 알아볼 듯한 변형 |
| original seed | recursive output | drift하지만 완전 붕괴 방지 |
| endpoint checkpoint | collapse midpoint | hollow layer |

가장 중요한 규칙:

> lower structure는 보존하고, upper detail부터 손상한다.

---

## 3.2 랜덤성은 white noise가 아니라 slow drift로 만든다

나쁜 방식:

```python
z = z + np.random.randn(*z.shape) * noise_scale
```

좋은 방식:

```python
noise_t = alpha * noise_t_minus_1 + (1 - alpha) * random_noise
z = z + noise_scale * noise_t
```

즉, 매 프레임 튀는 white noise가 아니라, latent 공간에서 천천히 미끄러지는 Brownian drift / smoothed random walk를 사용한다.

---

## 3.3 global destruction보다 local corruption

나쁜 방식:

```text
전체 token time-axis shuffle
전체 quantizer corruption
전체 checkpoint interpolation 중점 사용
```

좋은 방식:

```text
짧은 window 내부 token shuffle
upper quantizer만 corruption
checkpoint endpoint 근처 또는 collapse layer만 얇게 사용
```

---

## 3.4 dry/wet anchor는 필수

damage unit 뒤에는 반드시 dry/wet mix를 둔다.

```python
final = dry * (1 - wet) + damaged * wet
```

추천 범위:

| 상황 | dry_wet |
|---|---|
| subtle texture layer | 0.20 ~ 0.40 |
| main processed sound | 0.40 ~ 0.70 |
| collapse / transition | 0.70 ~ 1.00 |
| MVP-D collapse layer | 0.05 ~ 0.20 |

---

## 3.5 metric-based Texture Governor를 둔다

각 chunk마다 output metric을 계산해서 다음 상태를 감지한다.

| Metric | 의미 |
|---|---|
| RMS too low | silence/collapse |
| RMS too high | explosion/clipping |
| spectral flatness high | white-noise-like |
| centroid too high | harsh/fizz |
| flux too low | boring static drone |
| ZCR too high | noisy crackle |

일단 1차 구현에서는 RMS, spectral flatness, centroid, ZCR만 사용해도 충분하다.

---

# 4. 제안된 핵심 패치 개요

이번 feedback_1에서 제안하는 구현은 다음 여섯 가지다.

```text
1. src/core/mix.py 추가
   - dry/wet anchoring
   - RMS matching
   - soft limiter

2. src/core/texture_metrics.py 추가
   - RMS
   - spectral flatness
   - spectral centroid
   - zero crossing rate

3. src/core/texture_governor.py 추가
   - metric 기반 wet 자동 감쇠

4. MVP-A latent perturb 수정
   - white noise 외 smoothed latent drift 추가

5. MVP-C token bending 수정
   - auto upper quantizer selection
   - local shuffle window 추가

6. MVP-A/C/D render pipeline 수정
   - damage output 생성 후 governor + dry/wet mix 적용
```

---

# 5. 새 파일 1 — `src/core/mix.py`

## 목적

망가진 output이 너무 튀거나, 너무 조용하거나, 원본과 완전히 단절되는 것을 막는다.

## 파일 내용

```python
"""Dry/wet anchoring and safe output mixing.

These utilities keep damage-based processors near the input manifold instead of
letting them drift into pure noise, clipping, or silence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixConfig:
    dry_wet: float = 1.0          # 0=dry input, 1=fully damaged output
    rms_match: bool = False       # match damaged RMS to dry RMS before mixing
    limiter: bool = True          # soft-limit after mixing
    limiter_drive: float = 1.0    # >1 increases saturation before tanh


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    """Return root-mean-square amplitude."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32))) + eps))


def match_rms(reference: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale target so its RMS roughly matches reference RMS."""
    ref = rms(reference, eps=eps)
    tgt = rms(target, eps=eps)
    if tgt <= eps:
        return target.astype(np.float32)
    return (target * (ref / tgt)).astype(np.float32)


def soft_limiter(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh limiter that prevents explosive damaged outputs."""
    drive = max(float(drive), 1e-6)
    y = np.tanh(x.astype(np.float32) * drive) / np.tanh(drive)
    return y.astype(np.float32)


def dry_wet_mix(
    dry: np.ndarray,
    wet_audio: np.ndarray,
    cfg: MixConfig,
    *,
    override_wet: float | None = None,
) -> np.ndarray:
    """Blend clean and damaged signals with optional RMS matching/limiting."""
    n = min(len(dry), len(wet_audio))
    if n == 0:
        return wet_audio.astype(np.float32)

    dry_n = dry[:n].astype(np.float32)
    wet_n = wet_audio[:n].astype(np.float32)

    if cfg.rms_match:
        wet_n = match_rms(dry_n, wet_n)

    wet = cfg.dry_wet if override_wet is None else override_wet
    wet = float(np.clip(wet, 0.0, 1.0))
    mixed = (1.0 - wet) * dry_n + wet * wet_n

    if len(wet_audio) > n:
        mixed = np.concatenate([mixed, wet_audio[n:].astype(np.float32)])

    if cfg.limiter:
        mixed = soft_limiter(mixed, drive=cfg.limiter_drive)
    return mixed.astype(np.float32)
```

## 해설

- `dry_wet`: 원본과 손상본의 비율.
- `rms_match`: 손상본 RMS를 원본에 맞춰서 갑작스러운 amplitude jump 방지.
- `soft_limiter`: harsh explosion/clipping 방지.
- `override_wet`: Texture Governor가 chunk별로 wet을 줄일 수 있게 하기 위한 인자.

---

# 6. 새 파일 2 — `src/core/texture_metrics.py`

## 목적

손상된 output이 다음 상태로 갔는지 감지한다.

- 노이즈화
- silence collapse
- 고역 fizz
- 불안정한 crackle

## 파일 내용

```python
"""Small numpy-only audio metrics for detecting collapse/noise drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextureMetrics:
    rms: float
    spectral_flatness: float
    spectral_centroid_hz: float
    zero_crossing_rate: float


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = audio.astype(np.float32).reshape(-1)
    if len(x) < frame_size:
        x = np.pad(x, (0, frame_size - len(x)))
    n_frames = 1 + max(0, (len(x) - frame_size) // hop_size)
    frames = np.stack([x[i * hop_size : i * hop_size + frame_size] for i in range(n_frames)])
    return frames


def compute_texture_metrics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 2048,
    hop_size: int = 1024,
    eps: float = 1e-8,
) -> TextureMetrics:
    """Compute metrics that distinguish texture from noise/silence."""
    x = audio.astype(np.float32).reshape(-1)
    if x.size == 0:
        return TextureMetrics(0.0, 0.0, 0.0, 0.0)

    rms_val = float(np.sqrt(np.mean(x * x) + eps))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.float32))))) if len(x) > 1 else 0.0

    frames = _frame_audio(x, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) + eps
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate)).astype(np.float32)

    flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    centroid = np.sum(spec * freqs[None, :], axis=1) / np.sum(spec, axis=1)

    return TextureMetrics(
        rms=rms_val,
        spectral_flatness=float(np.mean(flatness)),
        spectral_centroid_hz=float(np.mean(centroid)),
        zero_crossing_rate=zcr,
    )
```

## 해설

### RMS

너무 낮으면 collapse/silence, 너무 높으면 clipping/explosion.

### Spectral flatness

값이 높을수록 white-noise-like하다.  
일반적으로 tonal/structured sound는 flatness가 낮고, broadband noise는 flatness가 높다.

### Spectral centroid

너무 높으면 고역 fizz / harshness 가능성이 높다.

### Zero Crossing Rate

과하게 높으면 crackle, harsh noise, HF artifact일 가능성이 있다.

---

# 7. 새 파일 3 — `src/core/texture_governor.py`

## 목적

metric을 보고 현재 chunk의 wet 값을 자동으로 줄인다.

이 모듈은 audio를 “깨끗하게” 만들기 위한 게 아니다.  
목표는 damage를 보존하되, usable range 밖으로 나가지 않게 하는 것이다.

## 파일 내용

```python
"""Output-side guardrail for damage-based audio processors.

The governor does not try to make audio clean. It keeps the damaged signal from
becoming unusable pure noise, silence, or clipping by reducing wetness per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.texture_metrics import TextureMetrics, compute_texture_metrics


@dataclass
class TextureGovernorConfig:
    enabled: bool = False
    min_wet: float = 0.10
    flatness_max: float = 0.55
    rms_min: float = 1e-4
    rms_max: float = 0.85
    centroid_max_ratio: float = 0.42


@dataclass
class GovernorDecision:
    wet: float
    metrics: TextureMetrics
    reason: str


def govern_wet(
    damaged: np.ndarray,
    sample_rate: int,
    base_wet: float,
    cfg: TextureGovernorConfig,
) -> GovernorDecision:
    """Return a safer wet value for the current damaged chunk."""
    metrics = compute_texture_metrics(damaged, sample_rate)
    wet = float(np.clip(base_wet, 0.0, 1.0))
    reasons: list[str] = []

    if not cfg.enabled:
        return GovernorDecision(wet=wet, metrics=metrics, reason="disabled")

    if metrics.spectral_flatness > cfg.flatness_max:
        wet *= 0.55
        reasons.append("flatness")

    if metrics.rms < cfg.rms_min:
        wet *= 0.35
        reasons.append("silence")

    if metrics.rms > cfg.rms_max:
        wet *= 0.60
        reasons.append("rms_high")

    nyquist = sample_rate / 2.0
    if nyquist > 0 and metrics.spectral_centroid_hz / nyquist > cfg.centroid_max_ratio:
        wet *= 0.75
        reasons.append("centroid")

    wet = float(np.clip(wet, cfg.min_wet, base_wet))
    return GovernorDecision(wet=wet, metrics=metrics, reason="+".join(reasons) or "ok")
```

## 해설

예를 들어 원래 `dry_wet=0.40`인데 output이 white-noise-like하다고 판단되면 wet을 0.22 정도로 줄인다.  
즉, 다음과 같은 구조가 된다.

```text
damaged output 생성
→ metric 계산
→ 너무 위험하면 wet 감소
→ dry/wet mix
→ limiter
→ 저장/출력
```

---

# 8. MVP-A 구현 변경

## 8.1 목표

MVP-A는 RAVE latent `z`를 perturb해서 timbral drift를 만드는 모듈이다.

현재 문제는 Gaussian white noise를 바로 latent에 더하면, 일정 threshold 이상에서 **단순 fizz / identity collapse**로 가기 쉽다는 점이다.

따라서 다음 기능을 추가한다.

```text
noise_mode: white | smoothed
noise_smooth: 0.98
```

---

## 8.2 `src/modules/mvp_a/latent_perturb.py` 수정

### dataclass에 필드 추가

```python
@dataclass
class PerturbParams:
    noise_scale: float = 0.0
    dim_dropout: float = 0.0
    dim_shuffle: bool = False
    bias_vector: np.ndarray | None = None
    freeze_mask: np.ndarray | None = None
    noise_mode: str = "white"      # white | smoothed
    noise_smooth: float = 0.98
```

### noise 처리 변경

기존:

```python
if params.noise_scale > 0:
    out = out + rng.normal(0, params.noise_scale, size=out.shape).astype(out.dtype)
```

변경:

```python
if params.noise_scale > 0:
    noise = rng.normal(0, 1.0, size=out.shape).astype(out.dtype)

    if params.noise_mode == "smoothed":
        alpha = float(np.clip(params.noise_smooth, 0.0, 0.999))
        for ti in range(1, noise.shape[-1]):
            noise[:, :, ti] = alpha * noise[:, :, ti - 1] + (1.0 - alpha) * noise[:, :, ti]
        noise = noise / (np.std(noise) + 1e-6)

    elif params.noise_mode != "white":
        raise ValueError(f"unknown noise_mode: {params.noise_mode}")

    out = out + params.noise_scale * noise
```

---

## 8.3 MVP-A config 추천값

`experiments/mvp_a_rave_latent/config.yaml`

```yaml
perturb:
  noise_scale: 0.08
  noise_mode: smoothed
  noise_smooth: 0.98
  dim_dropout: 0.05
  dim_shuffle: false
  bias_vector: null
  freeze_mask: null

mix:
  dry_wet: 0.35
  rms_match: true
  limiter: true
  limiter_drive: 1.0

governor:
  enabled: true
  min_wet: 0.10
  flatness_max: 0.55
  rms_min: 0.0001
  rms_max: 0.85
  centroid_max_ratio: 0.42
```

---

## 8.4 MVP-A render pipeline 수정

`src/modules/mvp_a/render.py`에서 decode 후 바로 저장하지 말고 다음을 적용한다.

```python
out = decode(handle, z_pert)
decision = govern_wet(out, handle.sample_rate, cfg.mix.dry_wet, cfg.governor)
out = dry_wet_mix(chunk, out, cfg.mix, override_wet=decision.wet)
```

로그도 다음처럼 남긴다.

```python
logger.info(
    "chunk %d: z=%s, delta_rms=%.4f, wet=%.3f, flat=%.3f, rms=%.4f, guard=%s",
    idx, z.shape, delta, decision.wet,
    decision.metrics.spectral_flatness, decision.metrics.rms, decision.reason,
)
```

---

# 9. MVP-C 구현 변경

## 9.1 목표

MVP-C는 EnCodec/DAC의 RVQ token을 조작하는 모듈이다.

현재 가장 중요한 문제는 이것이다.

> `quantizer_range: null`일 때 전체 quantizer를 손상하면 lower/coarse structure까지 무너져 identity collapse로 갈 가능성이 크다.

따라서 기본값을 다음으로 바꾼다.

```text
explicit quantizer_range가 없으면 upper/detail RVQ layer만 자동 선택
```

---

## 9.2 `BendParams` 수정

`src/modules/mvp_c/token_bend.py`

```python
@dataclass
class BendParams:
    mode: str
    rate: float
    quantizer_range: tuple[int, int] | None = None
    codebook_size: int = 1024
    invalid_token: int = -1
    shuffle_window: int | None = None
```

---

## 9.3 quantizer range normalization

```python
lo, hi = params.quantizer_range or (0, n_q)

if lo < 0:
    lo = max(0, n_q + lo)
if hi <= 0:
    hi = max(0, n_q + hi)

lo = int(np.clip(lo, 0, n_q))
hi = int(np.clip(hi, lo + 1, n_q))

sub = out[lo:hi]
```

이렇게 하면 `[-8, 0]` 같은 range도 뒤쪽 quantizer를 뜻하도록 확장할 수 있다.

---

## 9.4 local shuffle 구현

기존 global shuffle은 시간 구조를 너무 쉽게 깨뜨린다.

기존:

```python
n = max(1, int(params.rate * t))
cols = rng.choice(t, size=n, replace=False)
shuffled = cols.copy()
rng.shuffle(shuffled)
sub[:, cols] = sub[:, shuffled]
```

변경:

```python
elif params.mode == "shuffle":
    # Global time shuffling destroys continuity quickly.
    # If shuffle_window is set, only shuffle inside short local windows.
    if params.shuffle_window is not None and params.shuffle_window > 1:
        win = int(params.shuffle_window)

        for start in range(0, t, win):
            end = min(t, start + win)
            width = end - start
            if width <= 1:
                continue

            n = max(1, int(params.rate * width))
            cols = start + rng.choice(width, size=min(n, width), replace=False)
            shuffled = cols.copy()
            rng.shuffle(shuffled)
            sub[:, cols] = sub[:, shuffled]

    else:
        n = max(1, int(params.rate * t))
        cols = rng.choice(t, size=n, replace=False)
        shuffled = cols.copy()
        rng.shuffle(shuffled)
        sub[:, cols] = sub[:, shuffled]
```

목표 변화:

```text
global shuffle → temporal collapse
local shuffle  → granular stutter / micro-scramble
```

---

## 9.5 auto upper quantizer selection

`experiments/mvp_c_encodec_bend/main.py`에 추가한다.

```python
def _finalize_params(params: BendParams, cfg: dict, handle) -> BendParams:
    auto_upper = cfg["bend"].get("auto_upper_fraction")

    if params.quantizer_range is None and auto_upper is not None:
        frac = float(auto_upper)
        lo = max(
            0,
            min(
                handle.n_quantizers - 1,
                int(round(handle.n_quantizers * (1.0 - frac))),
            ),
        )
        params.quantizer_range = (lo, handle.n_quantizers)

    elif params.quantizer_range is not None:
        lo, hi = params.quantizer_range

        if lo < 0:
            lo = handle.n_quantizers + lo
        if hi <= 0:
            hi = handle.n_quantizers + hi

        params.quantizer_range = (max(0, lo), min(handle.n_quantizers, hi))

    return params
```

---

## 9.6 MVP-C config 추천값

`experiments/mvp_c_encodec_bend/config.yaml`

```yaml
bend:
  mode: bit_flip
  rate: 0.03
  quantizer_range: null
  auto_upper_fraction: 0.33
  shuffle_window: 12
  codebook_size: 1024

mix:
  dry_wet: 0.40
  rms_match: true
  limiter: true
  limiter_drive: 1.0

governor:
  enabled: true
  min_wet: 0.10
  flatness_max: 0.55
  rms_min: 0.0001
  rms_max: 0.85
  centroid_max_ratio: 0.42
```

---

## 9.7 MVP-C render pipeline 수정

`src/modules/mvp_c/render.py`

```python
out = decode_tokens(handle, bent)
decision = govern_wet(out, handle.sample_rate, cfg.mix.dry_wet, cfg.governor)
out = dry_wet_mix(chunk, out, cfg.mix, override_wet=decision.wet)
```

로그:

```python
logger.info(
    "chunk %d: in=%d tokens, bent=%d diffs, wet=%.3f, flat=%.3f, rms=%.4f, guard=%s",
    idx, tokens.size, int((bent != tokens).sum()), decision.wet,
    decision.metrics.spectral_flatness, decision.metrics.rms, decision.reason,
)
```

---

# 10. MVP-D 구현 변경

## 10.1 목표

MVP-D는 checkpoint morphing을 목표로 했지만, 현재 관찰상 독립적으로 학습된 RAVE checkpoint 사이의 naive interpolation은 중간 영역에서 near-silence collapse를 일으킨다.

따라서 MVP-D는 다음처럼 재정의하는 게 좋다.

```text
checkpoint morphing synthesizer
→ checkpoint collapse / hollow layer synthesizer
```

즉, D를 메인 소스로 쓰는 것이 아니라 얇은 ghost layer로 쓴다.

---

## 10.2 MVP-D config 추천값

`experiments/mvp_d_ckpt_morph/config.yaml`

```yaml
perturb:
  noise_scale: 0.05
  noise_mode: smoothed
  noise_smooth: 0.98
  dim_dropout: 0.10
  dim_shuffle: false
  bias_vector: null
  freeze_mask: null

mix:
  dry_wet: 0.15
  rms_match: true
  limiter: true
  limiter_drive: 1.0

governor:
  enabled: true
  min_wet: 0.05
  flatness_max: 0.55
  rms_min: 0.0001
  rms_max: 0.85
  centroid_max_ratio: 0.42
```

---

## 10.3 MVP-D 사용법

좋은 사용:

```text
t = 0.000 ~ 0.020 endpoint 근처
t = 0.980 ~ 1.000 endpoint 근처
dry_wet = 0.05 ~ 0.20
```

위험한 사용:

```text
t = 0.10 ~ 0.90
dry_wet = 1.0
MVP-D 단독 메인 출력
```

권장 사용:

```text
dry input
+ MVP-A latent perturb texture
+ MVP-C token bending grit
+ MVP-D collapse ghost layer
```

예시 mix:

```text
final =
  0.35 * dry_input
+ 0.35 * MVP-A_output
+ 0.20 * MVP-C_output
+ 0.10 * MVP-D_collapse_texture
```

---

# 11. config loader 수정 패턴

MVP-A/C/D의 `main.py`에는 공통적으로 `_build_render_config` 함수를 추가한다.

```python
def _build_render_config(cfg: dict) -> RenderConfig:
    render_cfg = cfg.get("render", {})
    mix_cfg = cfg.get("mix", {})
    gov_cfg = cfg.get("governor", {})

    return RenderConfig(
        chunk_seconds=float(render_cfg.get("chunk_seconds", 4.0)),
        overlap_seconds=float(render_cfg.get("overlap_seconds", 0.05)),
        mix=MixConfig(
            dry_wet=float(mix_cfg.get("dry_wet", 1.0)),
            rms_match=bool(mix_cfg.get("rms_match", False)),
            limiter=bool(mix_cfg.get("limiter", True)),
            limiter_drive=float(mix_cfg.get("limiter_drive", 1.0)),
        ),
        governor=TextureGovernorConfig(
            enabled=bool(gov_cfg.get("enabled", False)),
            min_wet=float(gov_cfg.get("min_wet", 0.10)),
            flatness_max=float(gov_cfg.get("flatness_max", 0.55)),
            rms_min=float(gov_cfg.get("rms_min", 1e-4)),
            rms_max=float(gov_cfg.get("rms_max", 0.85)),
            centroid_max_ratio=float(gov_cfg.get("centroid_max_ratio", 0.42)),
        ),
    )
```

필요 import:

```python
from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
```

---

# 12. 실험 프리셋

## 12.1 Preset 1 — 안전한 texture

목표: 원본 구조를 확실히 유지한 채 표면만 흔들기.

```yaml
MVP-A:
  perturb:
    noise_scale: 0.08
    noise_mode: smoothed
    noise_smooth: 0.98
    dim_dropout: 0.05
    dim_shuffle: false
  mix:
    dry_wet: 0.35

MVP-C:
  bend:
    mode: bit_flip
    rate: 0.03
    auto_upper_fraction: 0.33
  mix:
    dry_wet: 0.40
```

예상 결과:

```text
원본의 body/rhythm은 유지
표면에 glassy grit / neural shimmer 추가
```

---

## 12.2 Preset 2 — 살아있는 glitch

목표: stutter를 만들되 완전 시간 붕괴는 피하기.

```yaml
MVP-C:
  bend:
    mode: shuffle
    rate: 0.05
    auto_upper_fraction: 0.50
    shuffle_window: 12
  mix:
    dry_wet: 0.45
```

예상 결과:

```text
global time scramble이 아니라 local granular stutter
```

---

## 12.3 Preset 3 — hollow ghost layer

목표: MVP-D collapse를 얇은 layer로 사용.

```yaml
MVP-D:
  morph:
    t: 0.005
  perturb:
    noise_scale: 0.05
    noise_mode: smoothed
    dim_dropout: 0.10
  mix:
    dry_wet: 0.15
```

예상 결과:

```text
원본 뒤에 비어 있는 신경망 잔향 / hollow texture layer
```

---

## 12.4 Preset 4 — collapse transition

목표: 상시 사운드가 아니라 transition/drop 용도.

```yaml
MVP-D:
  morph:
    t: 0.02 → 0.10 sweep

MVP-C:
  bend:
    mode: invalid_token
    rate: 0.01
    auto_upper_fraction: 0.25

mix:
  dry_wet: 0.70 temporarily
```

예상 결과:

```text
순간적으로 모델이 무너지는 gesture
```

---

# 13. 실행 명령

## 13.1 문법 검사

```bash
python -m compileall -q src experiments
```

## 13.2 seed audio 생성

```bash
python - <<'PY'
import numpy as np
import soundfile as sf
import os

sr = 48000
t = np.linspace(0, 5.0, sr * 5, endpoint=False)

sig = (
    0.4 * np.sin(2 * np.pi * 220 * t)
    + 0.3 * np.sin(2 * np.pi * 440 * t)
    + 0.2 * np.sin(2 * np.pi * 880 * t) * np.sin(2 * np.pi * 3 * t)
    + 0.05 * np.random.default_rng(0).standard_normal(len(t))
)

sig = (sig / np.max(np.abs(sig)) * 0.9).astype("float32")

os.makedirs("runs", exist_ok=True)
sf.write("runs/seed_48k.wav", sig, sr)
PY
```

## 13.3 MVP-A render

```bash
bash experiments/mvp_a_rave_latent/run.sh \
  --mode render \
  --input runs/seed_48k.wav \
  --output runs/mvp_a_guarded.wav
```

## 13.4 MVP-C render

```bash
bash experiments/mvp_c_encodec_bend/run.sh \
  --mode render \
  --input runs/seed_48k.wav \
  --output runs/mvp_c_guarded.wav
```

## 13.5 MVP-D render

```bash
bash experiments/mvp_d_ckpt_morph/run.sh \
  --mode render \
  --input runs/seed_48k.wav \
  --output runs/mvp_d_guarded.wav
```

---

# 14. 로그 확인 기준

render 후 log에서 다음 값들을 확인한다.

```text
wet
flat
rms
guard
```

예시:

```text
chunk 0: wet=0.400, flat=0.211, rms=0.1320, guard=ok
chunk 1: wet=0.220, flat=0.684, rms=0.1800, guard=flatness
chunk 2: wet=0.140, flat=0.300, rms=0.0000, guard=silence
```

해석:

| guard | 의미 |
|---|---|
| ok | 안전 범위 |
| flatness | white-noise-like해서 wet 감소 |
| silence | collapse/silence라 wet 감소 |
| rms_high | clipping/explosion 위험 |
| centroid | harsh/fizz 고역 치우침 |

---

# 15. 실험 기록 양식

각 실험은 최소한 다음처럼 기록한다.

```json
{
  "experiment_id": "mvp_c_upper_bitflip_003",
  "input": "runs/seed_48k.wav",
  "module": "mvp_c",
  "params": {
    "mode": "bit_flip",
    "rate": 0.03,
    "auto_upper_fraction": 0.33,
    "dry_wet": 0.40,
    "governor": true
  },
  "observed": {
    "identity_preserved": true,
    "noise_like": false,
    "texture_quality": "glassy grit",
    "usable_as": "background layer / transition"
  },
  "notes": "dominant partials retained; high-frequency shimmer added"
}
```

---

# 16. 다음 단계 제안

## 16.1 Texture Governor v2

현재 governor는 wet만 줄인다.  
다음 버전에서는 module parameter도 직접 조절할 수 있다.

예:

```text
flatness high → MVP-C rate 감소
rms too low → dry_wet 감소, seed injection 증가
centroid high → upper quantizer range 축소
flux too low → latent drift 증가
```

구조:

```python
@dataclass
class GovernorAction:
    wet: float
    rate_scale: float
    noise_scale_delta: float
    force_seed_injection: bool
```

---

## 16.2 Onset-aware corruption

전체 구간을 균일하게 망가뜨리지 말고 onset/sustain을 나눈다.

```text
onset detected → short local shuffle
sustain → upper quantizer bit_flip
silence → no corruption
transition → invalid_token burst
```

이러면 noise가 아니라 gesture가 된다.

---

## 16.3 Token manifold-aware corruption

완전 random token 대신 다음을 사용한다.

```text
same codebook 내 nearby token
same cluster token
same temporal neighborhood token
same RMS/centroid region token
```

현재는 단순 bit_flip/drop/shuffle이지만, 다음 단계에서는 “비슷하지만 틀린 token”으로 바꾸는 것이 더 좋다.

---

## 16.4 Feedback loop에 seed reinjection

recursive feedback은 반드시 seed anchor를 유지한다.

```python
next_input = (
    0.65 * previous_output
    + 0.25 * original_seed
    + 0.10 * external_input
)
```

seed reinjection이 없으면 결국 다음 중 하나로 간다.

```text
silence
harsh noise
static drone
```

---

## 16.5 Max/MSP control mapping

추천 controller mapping:

| Control | Range | Target |
|---|---:|---|
| damage | 0~1 | MVP-C rate |
| hallucination | 0~1 | MVP-A noise_scale |
| anchor | 0~1 | dry/wet inverse |
| collapse | 0~1 | MVP-D t endpoint distance |
| grit | 0~1 | upper quantizer fraction |
| memory | 0~1 | feedback amount |
| safety | 0~1 | governor strength |

Max/MSP 쪽에서는 “damage”와 “anchor”를 반대로 묶으면 좋다.

```text
damage ↑ → wet ↑
anchor ↑ → wet ↓
safety ↑ → governor stronger
```

---

# 17. 최종 결론

현재 AudioArt repo는 이미 “잘 깨부수는 장치”를 갖고 있다.  
다음 단계에서 필요한 것은 “잘 안 죽게 붙잡는 장치”다.

따라서 feedback_1의 핵심은 다음이다.

```text
1. lower/coarse structure는 보존한다.
2. upper/detail layer만 먼저 손상한다.
3. white noise 대신 smoothed latent drift를 사용한다.
4. global shuffle 대신 local shuffle을 사용한다.
5. dry/wet anchor를 모든 MVP output 뒤에 둔다.
6. RMS/flatness/centroid 기반 Texture Governor를 둔다.
7. MVP-D collapse는 메인 소스가 아니라 얇은 ghost layer로 쓴다.
```

한 문장으로 정리하면:

> 노이즈가 아니라 텍스쳐가 되려면,  
> 모델이 완전히 못 알아듣게 만들면 안 되고  
> **겨우 오해할 수 있는 정도로만 괴롭혀야 한다.**

---

# 18. 이후 구현 보고 (Post-Implementation, 2026)

feedback_1 작성 당시에는 4개의 MVP (A, B, C, D) 만 존재했다. 그 이후 본 설계 원리에 따라 다음 항목들이 실제로 구현되었다.

## 18.1 새 MVP 5종 (E ~ I)

원래 MVP-A 의 latent perturbation 한 갈래만 가지고 있던 RAVE 도메인이 다섯 종의 변종으로 분화했다. 모두 anchored corruption 원리를 그대로 따라간다 — *원본 구조는 잡아둔 채 표면만 변형*.

| MVP | 손상 모드 | feedback_1 원칙과의 연결 |
|---|---|---|
| **E** Neural Granular | latent 메모리 버퍼에서 과거 grain 을 현재로 투사 | "local continuity" 보존 — 시간 표류를 점진적으로 |
| **F** Spectral Frozen | 상위 50% latent 차원을 주기적으로 동결 + crossfade | "upper/detail 만 손상" 원칙의 직접 구현 |
| **G** Latent Feedback | latent 공간 delay line 으로 재귀 에코 | "feedback entropy governor" — 자기 진화 메아리 |
| **H** Codebook Organ | 입력 없이 소수/Fibonacci 토큰 직접 생성 | drone bed 로 사용, 다른 버스의 anchor 역할 |
| **I** Bass Massive | 하위 RVQ 토큰 smear/jitter/fold | "lower/coarse structure 보존" 원칙의 부분 위배 — 단, sub-bass 영역만 손상시켜 다른 트랙의 80 Hz crossover 와 결합 시에는 결과적으로 anchored |

각 MVP 디렉토리에 자체 `results.json` 과 `README.md` 가 있고, 음상 sweep 결과까지 기록되어 있다.

## 18.2 Texture Guard 의 코드 실현

§13–§16 에서 제안된 텍스쳐 보호 장치는 `src/core/` 안에 다음 모듈로 구현되었다:

- **`mix.py`** — dry/wet anchor + RMS match + soft tanh limiter + 80 Hz crossover anchor (sub_boost_db 옵션 포함).
- **`texture_metrics.py`** — RMS / spectral flatness / spectral centroid / ZCR / NaN-Inf 감지.
- **`texture_governor.py`** — flatness > 0.55, RMS 범위 ([1e-4, 0.85]), centroid > Nyquist × 0.42 임계치 기반 청크별 자동 wet 감쇠. NaN 감지 시 emergency wet=0.

모든 MVP render 함수가 청크 단위로 이 가드를 통과한다. 운영 중 2:08 지점에서 일어났던 NaN 발생 끊김 사고가 이 가드로 해결됐다.

## 18.3 매크로 작곡 단계 — Multinet

9 개 MVP 를 묶는 5 개 매크로 네트워크가 `scripts/multinet.py` 안에 정의됐다:

| Net | 분류 | feedback_1 의 어느 원칙을 적용 |
|---|---|---|
| **Net 1** Crystal Cathedral | 5-bus 병렬 믹스 | 다층 동시 손상이지만 각 버스가 dw < 0.7 + 80 Hz 앵커로 묶임 |
| **Net 2** Recursive Organ | 3-pass 매크로 피드백 | feedback entropy governor 의 직접 시연 — pass 가 누적될수록 정체성 표류 |
| **Net 3** Decoding Chamber | 직선 9 단 손상 | 단계별 dw=0.45 + 80 Hz crossover 분기로 dry 앵커 — collapse 방지의 모범 사례 |
| **Net Max** Cathedral Hive | 8-bus + cross-feedback + 2-pass | 모든 9 개 MVP 통합 — anchored corruption 의 풀-스케일 적용 |
| **Net Dynamic** Tempest | 8 버스 + 시간 가변 envelope + filter sweep + 3 impulse 이벤트 | "정적 텍스쳐 → 작곡" 으로 격상 |

자세한 토폴로지: [`MULTINET_ARCHITECTURE.md`](MULTINET_ARCHITECTURE.md).

## 18.4 메타 작곡 단계 — Meta-Symphony

매크로넷의 네트워크. 4 개 매크로 출력을 3 분 타임라인 위에서 LFO crossfade (60s/45s) + 스테레오 드리프트 (20s/25s) + 100 Hz sub 재주입으로 엮어 stereo 최종 작품 생성. AudioArt 스택에서 첫 stereo 출력. 자세한 설계: [`META_SYMPHONY_ARCHITECTURE.md`](META_SYMPHONY_ARCHITECTURE.md).

## 18.5 검증 가능 데모

`scripts/build_demo.py` 가 모든 단계의 wav 와 waveform 썸네일을 한 페이지로 묶어 `demo.html` 로 출력. 52 트랙 + 9 섹션 + sticky nav. 외부 청취자가 한 번에 결과를 검증할 수 있는 형태.

## 18.6 결국 무엇이 작동했는가

feedback_1 의 7개 핵심 명제 중 실제로 가장 결정적이었던 것:

1. **80 Hz crossover + sub_boost** — 거의 모든 매크로넷에서 이 한 줄이 collapse 와 noise 표류를 같이 막는다.
2. **per-stage dw < 0.7** — 누적 손상 체인의 안정성을 보장.
3. **Texture Governor 의 청크별 자동 감쇠** — 외부에서 손을 대지 않아도 시스템이 알아서 깨끗한 채로 머문다.
4. **MVP-D collapse 의 endpoint-only 사용** — Re-Basin 까지 시도했으나 결국 cliff sweep 으로 운영 영역을 좁히는 게 가장 실용적.

가장 덜 작동했던 것:
- **shuffle_window 의 효과** — 미세 윈도우 안 셔플은 의도와 달리 큰 차이가 없었다. 토큰 도메인의 시간 해상도 (75 fps) 가 너무 거칠어, window=12 와 window=4 가 청각적으로 거의 동일.
- **MVP-B 의 의미적 표류** — stub backend 만 동작 가능했고, 실제 Qwen2-Audio + AudioLDM2 다운로드는 다음 단계로 미뤄졌다.

## 18.7 향후 작업 — AudioLLM 통합

프로젝트 이름 **"AudioLLM-Art"** 가 가리키는 다음 단계는 현재 stub 으로 동작하는 MVP-B 의 실제 활성화와, AudioLLM 을 다른 모든 매크로넷의 의미 조건화 신호로 확장하는 것이다:

1. **MVP-B 실제 백본 활성화** — Qwen2-Audio-7B-Instruct (캡션) + AudioLDM2 (TTA) 다운로드 (~18 GB) 후 stub 대체.
2. **AudioLLM-aware perturbation** — 캡션 결과 텍스트를 다른 MVP 의 파라미터로 매핑 (예: "metallic" → C.bend.rate↑, F.auto_upper_fraction↑).
3. **Semantic anchor** — 80 Hz sub-bass 가 *물리적* 닻이라면, AudioLLM 캡션은 *의미적* 닻이 될 수 있다. 청크의 캡션이 너무 멀어지면 wet 감쇠 — Semantic Governor.
4. **Multi-agent caption ensemble** — Qwen-Audio, SALMONN, Pengi 의 disagreement 를 새 noise 채널로.
5. **AudioLLM 자체 손상** — 캡션 모델의 텍스트 임베딩 공간에 직접 노이즈 주입, 의미 표류의 "결" 자체를 변형.

이 단계가 들어오면 feedback_1 의 7번 명제 "MVP-D collapse 는 thin ghost layer 로" 같은 식의 *어떻게 안 죽일 것인가* 원칙이, AudioLLM 도메인에 새로 다시 한 번 적용되어야 한다.

---
