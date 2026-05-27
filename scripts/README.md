# Scripts

AudioArt 생태계의 고수준 작곡 엔진, 매크로 오케스트레이션 도구, 마스터링 유틸리티 모음.

## 핵심 엔진

- **`meta_symphony.py`** — 작곡의 정점. Net 1, 2, 3, Dynamic 을 3분 타임라인 위에서 스테레오 드리프트 + LFO 로 엮는 "네트워크의 네트워크". 설계는 [`docs/META_SYMPHONY_ARCHITECTURE.md`](../docs/META_SYMPHONY_ARCHITECTURE.md).
- **`multinet.py`** — 매크로 토폴로지 정의 (Net 1 / 2 / 3 / Max / Dynamic) 및 복합 신호 흐름 처리. 9개 MVP 모듈 A–I를 합성. 설계는 [`docs/MULTINET_ARCHITECTURE.md`](../docs/MULTINET_ARCHITECTURE.md).
- **`build_demo.py`** — 정적 HTML 데모 페이지 + waveform 썸네일 생성. 출력은 `demo.html`. 트랙 manifest 를 스크립트 안에서 편집 후 재실행.
- **`make_stub_rave.py`** — API 검증용 untrained RAVE `.ts` 스켈레톤 생성 (다운로드 없이 파이프라인 테스트).

## 마스터링 / 향상

- **`pyloudnorm_mastering.py`** — BS.1770-4 기반 −12.0 LUFS 라우드니스 정규화. 최종 출력에 권장.
- **`hifi_enhancer.py`** — 8 kHz 이상에 spectral exciting + 14 kHz air boost. HD 뉴럴 텍스처용.
- **`enhance_masterpiece.py`, `enhance_masterpiece_v2.py`** — 마스터피스용 다단 향상 파이프라인.
- **`aggressive_mastering.py`** — 강력한 컴프레션 + 라우드니스 극대화.
- **`final_bass_pro_master.py`** — NaN hardening 내장 + 80 Hz 크로스오버 + 8 dB sub-boost 의 베이스 헤비 마스터링.

## 대용량 SoX 파이프라인 (1시간급 작품)

68분 이상 작품에서 Python 텐서가 메모리 병목이 될 때 디스크 스트리밍으로 우회:

- **`run_hfo_master_sox.sh`** — 디스크 직접 리샘플링 + 필터링.
- **`run_f0_octave_sox_master.sh`** — f0 정렬 멀티 옥타브 믹싱.
- **`finish_hyperchord.sh`** — 9-옥타브 foundation 드론을 공격적 loudness maximizing 으로 마무리.

자세한 도입 배경은 [`docs/IMPLEMENTATION_REPORT_V1.md`](../docs/IMPLEMENTATION_REPORT_V1.md) §7 참고.

## 작곡 / 합성 보조

- **`create_masterpiece.py`, `create_masterpiece_2.py`** — 다층 뉴럴 심포니 빌더.
- **`create_granular_masterpiece.py`** — 250 ms grain 단위로 8 트랙을 stochastic 교차.
- **`create_full_spectrum_masterpiece.py`** — −2 ~ +2 옥타브 강제 분산 배치.
- **`create_layered_hfo_master.py`** — 다층 누적 HFO 마스터링.
- **`create_defined_galaxy.py`** — Tukey 윈도우 적용 그래뉼러 재구성.
- **`create_1hr_masterpiece.py`** — 1시간 분량 단일 작품 합성.
- **`generate_f0_script.py`, `generate_foundation_hyperchord.py`, `generate_foundation_high_octaves.py`** — 시드 / 화성 기반층 생성기.

## 분석 / 검증

- **`check_metrics.py`** — 디렉토리 일괄 RMS · Flatness · NaN 스캔.
- **`analyze_long_file.py`** — 장시간 파일의 청크별 메트릭 분석.

## 1시간 체인 / 큐 관리

- **`gpu_task_manager.py`, `gpu_task_manager_resume.py`** — 장시간 작업 큐, 재시작 지원.
- **`run_1hr_chain.py`, `run_1hr_freeze.py`** — 1시간 분량 체인 / freeze 렌더링 러너.

## 실행 패턴

```bash
# Foreground
PYTHONPATH=. python scripts/meta_symphony.py

# Background + 로그 + PID
nohup python scripts/meta_symphony.py > runs/masterpiece/meta_symphony.log 2>&1 &
echo $! > runs/masterpiece/meta_symphony_pid.txt
tail -f runs/masterpiece/meta_symphony.log
```

## 향후 작업

- **`build_demo.py` 확장** — manifest 를 YAML 외부 파일로 분리, 다국어 라벨 / 트랙 태그 / 검색 기능.
- **AudioLLM 통합 마스터링** — 최종 곡에 캡션 모델을 돌려 자동 트랙 설명 텍스트 생성.
- **재현성 패키지** — 각 마스터피스 wav 와 함께 사용된 config + git hash + 모델 SHA 를 sidecar JSON 으로 묶는 빌더 스크립트.
