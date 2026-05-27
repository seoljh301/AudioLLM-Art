# MVP-B — Caption ↔ Text-to-Audio Recursive Loop

## 가설
`audio → caption → mutate text → TTA → audio` 를 반복하면 의미적 표류가 누적된다. 시스템이 소리를 언어를 통해 반복 "번역"하면서 언어적 편향과 생성 아티팩트가 쌓여, 출력이 시드와 흥미롭게 발산한다.

## 모델 의존성

- **Caption**: `Qwen/Qwen2-Audio-7B-Instruct` (또는 대안: SALMONN, Pengi)
- **TTA**: `cvssp/audioldm2` (또는 대안: Stable Audio, Tango)
- 모두 HuggingFace, 최초 사용 시 자동 다운로드.

## 예상 음상

| 단계 | 양상 |
|---|---|
| 1–2 | 시드의 알아볼 수 있는 변형 |
| 3–4 | 스타일적 표류, 정체성이 헐거워짐 |
| 5+ | 의미적 붕괴 — 캡션이 환각, TTA 가 불가능한 사운드 텍스처 생성 |

## Sweep 가능한 파라미터

- `loop.depth ∈ [2, 10]`
- `loop.text_mutation_prob ∈ [0, 1]`
- `tta.guidance_scale ∈ [1, 10]`
- `tta.duration_s` — 단계별 출력 길이

## 백엔드

| 역할 | `backend: stub` (기본) | `backend: qwen2_audio` / `audioldm2` |
|---|---|---|
| caption | 결정론적 spectral-stats → 형용사 문장 | Qwen2-Audio-7B-Instruct |
| tta | 텍스트 해시 시드 FM 합성 | AudioLDM2 |

Stub 은 다운로드 없이 루프 제어 흐름을 검증한다. 실제 백엔드는:

- Qwen2-Audio 약 14 GB + AudioLDM2 약 4 GB 다운로드 필요.
- V100 (SM 7.0) 은 torch 2.12 + cu130 지원 안 됨 — `torch<=2.4 + cu121` 핀 또는 최신 GPU 사용.
- 큰 transformer 모델 로드는 HF 의 임의 코드 실행을 동반 — 신뢰 가능한 repo 만.

전환은 `config.yaml` 편집:

```yaml
caption:
  backend: qwen2_audio
tta:
  backend: audioldm2
```

## 실행

### 오프라인 루프 (seed → N 단계 → out_dir)
```bash
bash run.sh --mode render --input runs/mvp_b/seed_16k.wav --output runs/mvp_b/loop_out
```
출력 디렉토리에 `step_00_seed.wav`, `step_01.wav` ... `step_NN.wav` 와 `transcript.json` 저장.

### OSC 서버
```bash
bash run.sh --mode serve
```

## OSC 인터페이스

| 주소 | 인자 | 동작 |
|---|---|---|
| `/mvp_b/start` | string in_path, string out_dir | 루프 실행, `/mvp_b/done <transcript_path>` 응답 |
| `/mvp_b/error` | string | 실행 실패 시 응답 |

## 노트
- 비동기 — stub 도 depth=6에 1초 미만; 실제 백엔드는 단계 당 수 초. Max 는 `/mvp_b/start` 발사 후 `/mvp_b/done` 대기 패턴 권장.
- 각 루프 단계는 config seed + `text_mutation_prob` 가 동일하면 결정론적.

## 향후 작업

- **실제 백엔드 활성화** — AudioArt 의 "AudioLLM" 정체성이 진짜로 작동하는 모듈. Qwen2-Audio + AudioLDM2 다운로드 후 stub 대체.
- **Multi-agent caption ensemble** — Qwen-Audio, SALMONN, Pengi 를 동시에 돌려 disagreement 정도를 새 노이즈 채널로 사용.
- **AudioLLM 자체 손상** — 캡션 모델 임베딩 공간에 noise 주입, 의미 표류의 "결" 을 의도적으로 변질.
- **다른 매크로넷의 파라미터 사상** — 캡션 결과 텍스트를 C / I 의 손상 강도, F 의 freeze 비율 등으로 매핑하는 `net_semantic`.
- **번역 체인** — 캡션 → 한국어 번역 → 다시 영어 → TTA. 의미를 추가로 비틀기.
