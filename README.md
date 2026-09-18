# HAI Benchmark

진동 진단 모델을 같은 규칙으로 비교합니다. 세션이 둘이고, 순위는 **세션에 속한 분할의 평균**입니다.

| 세션 | 무엇을 재나 | 분할 |
|---|---|---|
| **일반화** | 학습에 없던 레일·미래 구간으로 옮겼을 때 | Hold A · Hold B · Hold C · Time |
| **Few-shot** | 학습 데이터가 적을 때 | 학습 1 · 2 · 5 · 10 · 20 % |

**리더보드 → https://97yong.github.io/hai-benchmark/**

데이터는 각자 로컬에 있습니다. 이 저장소에는 지표만 오갑니다.

## 쓰는 법

`submit.py` 한 파일만 받으면 됩니다. GitHub 계정만 있으면 되고, 따로 권한을 받을 필요는 없습니다.

```bash
gh auth login     # 처음 한 번만

python3 submit.py 홍길동 "CNN2D" pred_*.csv --input STFT --ckpt model.pt
```

채점은 각자 컴퓨터에서 돌고, 결과는 Issue 로 올라가 자동으로 리더보드에 반영됩니다.
예측과 라벨은 전송되지 않습니다.

### 예측 파일

열 두 개면 됩니다. 시나리오는 **행 번호로 자동 판별**하므로 파일명 규칙이 없습니다.

```csv
row,pred
12,1
13,1
```

| 값 | 클래스 |
|---|---|
| 0 | 정상 |
| 1 | 축계 |
| 2 | 기어계 |

`row` 는 `oht_loader` manifest 의 시험 행 번호입니다.

```python
from oht_loader import OHTData, load_manifest
d  = OHTData("<캐시 폴더>")
sp = load_manifest(d, "oht_loader_v2/splits/hold_C.json")
rows = sp.test                      # 이 순서대로 예측을 내면 됩니다
```

### 인자

| 인자 | 설명 |
|---|---|
| `--input` | 입력 표현. 예: `STFT`, `FFT`, `Raw` |
| `--split` | 분할을 직접 지정. **Few-shot 에 필요**합니다. 예: `--split time_sub05` |
| `--ckpt` | 체크포인트 경로. 파라미터 수를 세어 넣습니다 (PyTorch) |
| `--params` | 파라미터 수를 직접 적을 때. 예: `3.6M` |
| `--dry` | 등록 없이 점수만 확인 |
| `--loader` · `--cache` | 경로를 못 찾을 때. 환경변수 `OHT_LOADER` · `OHT_CACHE` 도 됩니다 |

## 채점 규약

| 항목 | 내용 |
|---|---|
| 순위 | 고른 세션에 속한 분할의 평균 (일반화 4개 · Few-shot 5개) |
| 집계 | 창 단위로 내면 채점기가 **주행 단위 다수결**로 묶습니다 |
| FNR | 고장을 정상이라 놓친 주행 비율 |
| FPR | 정상을 고장이라 부른 주행 비율 |
| FNR+FPR | 두 오류를 더한 값. 낮을수록 좋습니다 |
| 데이터 | 캐시 `content_id` 가 다르면 등록이 거부됩니다 |

## 일반화 세션

| 시나리오 | 학습 → 시험 |
|---|---|
| **Hold A** | B·C → A |
| **Hold B** | A·C → B |
| **Hold C** | A·B → C |
| **Time** | 앞 80% → 뒤 20% |

## Few-shot 세션

학습 데이터가 적을 때 얼마나 버티는지 재는 **학습 곡선** 세션입니다.
`oht_loader` v1.2.0 의 `subsample_train()` 으로 만든 manifest 를 씁니다.

| 분할 | 학습 | 시험 |
|---|---|---|
| `time_sub01` | 400창 (1%) | Time 과 **완전히 동일** (9,896창 · 1,331 주행) |
| `time_sub02` | 794창 (2%) | 〃 |
| `time_sub05` | 1,986창 (5%) | 〃 |
| `time_sub10` | 3,975창 (10%) | 〃 |
| `time_sub20` | 7,951창 (20%) | 〃 |

설계상 세 가지가 보장됩니다 — **시험셋 고정**(조건마다 y축이 달라지지 않게), **포함 관계**
(1% ⊂ 2% ⊂ 5% ⊂ 10% ⊂ 20%, 차이가 학습량 때문임이 보장되게), **셀 층화**(1% 에서도 36개 셀이
모두 살아남게).

```bash
python3 submit.py 홍길동 "CNN2D" pred.csv --split time_sub05
```

시험셋 행이 Time 과 같아 행 번호로는 가를 수 없으므로 `--split` 이 필요합니다.
파일명에 `time_sub05` 가 들어 있으면 생략해도 됩니다.
**다섯 조건을 다 내면** 리더보드 그래프에 학습 곡선이 완성됩니다.
