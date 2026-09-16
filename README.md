# HAI Benchmark

진동 진단 모델을 같은 규칙으로 비교합니다. 순위는 **네 시나리오 평균**입니다.

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
| `--ckpt` | 체크포인트 경로. 파라미터 수를 세어 넣습니다 (PyTorch) |
| `--params` | 파라미터 수를 직접 적을 때. 예: `3.6M` |
| `--dry` | 등록 없이 점수만 확인 |
| `--loader` · `--cache` | 경로를 못 찾을 때. 환경변수 `OHT_LOADER` · `OHT_CACHE` 도 됩니다 |

## 채점 규약

| 항목 | 내용 |
|---|---|
| 순위 | 네 시나리오 평균 |
| 집계 | 창 단위로 내면 채점기가 **주행 단위 다수결**로 묶습니다 |
| FNR | 고장을 정상이라 놓친 주행 비율 |
| FPR | 정상을 고장이라 부른 주행 비율 |
| FNR+FPR | 두 오류를 더한 값. 낮을수록 좋습니다 |
| 데이터 | 캐시 `content_id` 가 다르면 등록이 거부됩니다 |

## 시나리오

| 시나리오 | 학습 → 시험 | 시험 주행 | 정상 주행 |
|---|---|---|---|
| **Hold A** | B·C → A | 1,861 | 83 |
| **Hold B** | A·C → B | 960 | 80 |
| **Hold C** | A·B → C | 3,824 | 368 |
| **Time** | 앞 80% → 뒤 20% | 1,331 | 82 |

Hold A 는 정상 주행이 83개뿐이라 작은 차이는 우연일 수 있습니다.
