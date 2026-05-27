# Max/MSP 패치

`.maxpat` 패치를 이 디렉토리에 둔다. 9개 MVP 모듈 (A–I) 각각이 실시간 제어 / 원격 렌더링을 위한 OSC 포트 쌍을 갖는다.

## OSC 포트 맵

| MVP | 도메인 | Python listen (← Max) | Python send (→ Max) |
|:---:|:---|:---:|:---:|
| **A** | Latent Perturb | 7400 | 7401 |
| **B** | Caption Loop | 7410 | 7411 |
| **C** | Token Bending | 7420 | 7421 |
| **D** | Morphing | 7430 | 7431 |
| **E** | Neural Granular | 7440 | 7441 |
| **F** | Spectral Freeze | 7450 | 7451 |
| **G** | Latent Feedback | 7460 | 7461 |
| **H** | Codebook Organ | 7470 | 7471 |
| **I** | Bass Massive | 7480 | 7481 |

> 일부 실험 (`mvp_c_encodec_bend/config.yaml`, `mvp_d_ckpt_morph/config.yaml`) 은 5007/5008, 5009/5010 등 별도 포트를 쓰도록 사용자 정의되어 있으니, 각 `config.yaml` 의 `osc:` 블록을 확인.

## RAVE 통합

RAVE 기반 모듈 (A, D, E, F, G) 은 `nn~` 외부 객체로 Max 내부에서 네이티브 추론 가능:

- [nn~ GitHub (ACIDS-IRCAM)](https://github.com/acids-ircam/nn_tilde)

`.ts` 체크포인트를 `nn~` 에 직접 로드하고, 표준 Max 메시지로 latent 공간을 perturb 할 수 있다. EnCodec 기반 모듈 (C, H, I) 은 비동기 처리 트리거에 Python OSC bridge 를 사용한다.

> ⚠️ TorchScript `.ts` 파일은 임의 코드를 실행하니, 신뢰 가능한 출처에서만 다운로드 (예: [`acids-ircam/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)).

## 일반 메시지 패턴

각 MVP main.py 의 OSC 핸들러는 다음 명령을 받는다:

| 주소 | 인자 | 동작 |
|---|---|---|
| `/mvp_X/render` | 입력 wav 경로, 출력 wav 경로 | 비동기 렌더 트리거, 완료 시 `/mvp_X/done <out>` |
| `/mvp_X/<param>` | 값 | 라이브 파라미터 변경 (예: `/mvp_a/noise 0.3`) |
| `/mvp_X/error` | 에러 문자열 | 렌더 실패 시 Python → Max 응답 |

자세한 핸들러 목록은 각 `experiments/mvp_X_*/main.py` 와 `experiments/mvp_X_*/README.md` 참고.

## 향후 작업

- **메인 패치 템플릿 (`audioart_master.maxpat`)** — 9 MVP의 핸들을 한 곳에서 라우팅하는 컨트롤 표면.
- **AudioLLM-aware 핸들러** — MVP-B 의 실제 백본이 활성화되면 캡션 텍스트를 Max 의 [text] 객체로 회신, 시각적 모니터링 + 다른 패치에서 활용.
- **자동 매핑 패치** — `multinet.py` 의 토폴로지 정의를 OSC 컨트롤 매핑으로 자동 생성하는 헬퍼.
