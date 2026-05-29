# MVP-S — Random Prompt TTA

**위치:** matrix `∅ → audio · text`.

## 가설
텍스트 prompt 자체를 발원지로 두면 AudioLLM (또는 그 fallback 인 procedural Griffin-Lim) 이 무엇을 "기본 상상"하는지 들린다. 입력 없음, 텍스트만으로 합성.

## 백엔드
- `use_real: false` → procedural Griffin-Lim (MVP-P 의 키워드 매핑 활용)
- `use_real: true` → AudioLDM 시도 (diffusers 필요, 실패 시 자동 fallback)

## 사용 예
```bash
bash experiments/mvp_s_random_prompt/run.sh --out runs/mvp_s/random_concat_12s.wav
```
