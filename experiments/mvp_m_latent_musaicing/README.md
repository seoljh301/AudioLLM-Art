# MVP-M — Latent Granular Resynthesis (Musaicing)

**Paper:** Kui et al., *Latent Granular Resynthesis using Neural Audio Codecs*, ISMIR 2025 (arXiv:2507.19202)

## 가설

뉴럴 오디오 코덱의 연속 잠재 공간에서 직접 그래뉼러 합성을 수행하면, 파형 도메인 concatenative musaicing 보다 자연스러운 합성과 codec-induced 음색 변화를 동시에 얻을 수 있다. Training-free.

## 파이프라인

```
source corpus ──encoder──▶ latent (dim, T_s)
                          │
                          └──split (grain_size, stride)─▶ Grain Bank (N, dim, G)
                                                                │
target ──encoder──▶ latent ──split──▶ window features ──cos sim─┤
                                                                ▼
                                              softmax(τ) → categorical sample
                                                                │
                                                       indices → assemble
                                                                │
                                                      ──decoder──▶ output wav
```

## 주요 하이퍼파라미터

| 이름 | 의미 |
|---|---|
| `grain_size` | latent 프레임 단위 grain 크기. 8 ≈ 107 ms @ 75 Hz |
| `stride` | source corpus 윈도우 stride (작을수록 grain 많아짐) |
| `target_stride` | target 분할 stride; `grain_size` 와 같으면 겹침 없음 |
| `temperature` | softmax τ. 0=argmax(충실도↑), ∞=uniform(다양성↑) |
| `overlap_add` | grain 경계 cosine crossfade |
| `walk_strength` | >0 일 때 인접 corpus index 선호 (Markov 풍 temporal continuity) |
| `dry_wet` | 1=pure musaicing, 0=target passthrough |

## 사용 예

```bash
bash experiments/mvp_m_latent_musaicing/run.sh \
  --corpus data/corpus/*.wav \
  --target runs/sine_30s.wav \
  --out runs/mvp_m/match_001.wav
```

## 향후 (V2)

- Music2Latent / Stable Audio VAE 백엔드 추가
- Mimi 시멘틱 토큰 기반 매칭 (의미 단위 grain bank)
- AudioLLM 의 텍스트 임베딩으로 grain 선택 가이드
