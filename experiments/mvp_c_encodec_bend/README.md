# MVP-C — EnCodec / DAC Token Bending

## 가설
RVQ 토큰을 layer 별로 다르게 손상시키면 음악적으로 구분되는 아티팩트가 나온다:

- **상위 quantizer** — 미세한 residual detail 담당 → 손상 = 고주파 텍스처 손상.
- **하위 quantizer** — 거친 구조 담당 → 손상 = 정체성 붕괴.

layer 를 선택적으로 손상시키면 timbre 와 structure 를 독립 제어 가능하다.

## 모델 의존성

- `encodec` pip 패키지 (Meta EnCodec, 24 kHz / 48 kHz)
- `descript-audio-codec` (DAC 44.1 kHz용)

## 예상 음상

| 모드 | 효과 |
|---|---|
| `bit_flip` (낮은 rate, 상위 quantizer) | 유리 같은 granular grit |
| `quantizer_drop` | 텅 빈 codec-broken 텍스처 |
| `shuffle` | 시간적 stutter / scramble |
| `invalid_token` | 침묵 공동 + decoder 환각 |

## Sweep 가능한 파라미터

- `bend.mode ∈ {bit_flip, quantizer_drop, shuffle, invalid_token}`
- `bend.rate ∈ [0, 0.3]`
- `bend.quantizer_range` — 상위 vs 하위 분리 (음수 인덱스로 상위 지정 가능, 예: `(-3, 0)` 은 마지막 3개)
- `bend.shuffle_window` — shuffle 모드의 윈도우 크기 (리듬 구조 보존)
- `bend.auto_upper_fraction` — 상위 자동 선택 비율

## 실행

오프라인 렌더 (파일 → 손상 파일):
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

OSC 서버 (Max 에서 `/mvp_c/render <in> <out>` 트리거):
```bash
bash run.sh --mode serve
```

## OSC 인터페이스

| 주소 | 인자 | 동작 |
|---|---|---|
| `/mvp_c/rate` | float | 손상 비율 |
| `/mvp_c/mode` | string | 모드 전환 |
| `/mvp_c/render` | string in, string out | 렌더 트리거 |

## 하드웨어 노트
- V100 GPU: 루트 `environment.yaml` 의 `torch==2.4.1 + cu121` 핀 의존.
- V100 위 EnCodec 24 kHz: 5 초 오디오가 약 1.6 초 (CPU 대비 ≈ 170× 빠름).
- CPU 위 EnCodec 24 kHz: 약 50× 느림. 배치는 가능, 스트리밍은 빠듯.

## 안전장치 (자동 적용)
모든 청크는 dry/wet + soft limiter + texture governor 통과. 특히 `invalid_token` 모드에서 클램프 안 된 −1 토큰이 decoder 폭주를 일으킬 수 있어 NaN 가드가 핵심.

## 향후 작업

- **DAC 44.1 kHz 비교 sweep** — 코드북 기하구조가 다르므로 같은 손상이 어떻게 다른 아티팩트를 만드는지 정량 비교.
- **AudioLLM 조건부 손상** — 캡션 prompt 임베딩이 `quantizer_range` 와 `mode` 를 시간 가변으로 결정하는 prompt-conditioned bending.
- **시간 가변 손상 envelope** — `rate` 와 `shuffle_window` 를 LFO 또는 piecewise breakpoint 로 자동화, 곡 흐름과 동기화.
- **Mimi codec 지원** — Kyutai Mimi 의 12.5 fps semantic + acoustic 토큰 듀얼 스트림에서 두 종 토큰을 따로 손상시키는 변형.
