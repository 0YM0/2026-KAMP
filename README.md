# 2026 KAMP: YOLO11s + ROI 기반 이물질 탐지

색상 표시 제거와 인페인팅, YOLO11s 탐지, ROI 검증, NMS, 좌표 보정을 결합한 이물질 탐지 프로젝트다.

## 구성

| 항목 | 경로 |
|---|---|
| 소스코드 | `src/` |
| 설정 | `configs/` |
| 데이터 | `data/` |
| 학습된 모델 | `models/` |
| 검증 및 테스트 결과 | `results/` |
| 실행 환경 | `requirements.txt` |

데이터는 학습 360장, 검증 68장, 테스트 72장으로 분할되어 있다.

## 설치

Python 3.12 환경을 권장한다.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
# 테스트 및 검증 데이터 추론
python src/run.py test
python src/run.py validation

# 이미지 또는 폴더 추론
python src/run.py predict --input data/raw/images --output results/custom_predictions.json
```

입력 형식은 BMP, PNG, JPG, JPEG를 지원한다.

## 학습

```bash
python src/run.py preprocess
python src/train.py detector --device cpu
python src/train.py verifier
python src/train_shared_quality.py train
```

## 결과

| 세트 | 점수 기준 | IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| validation | 0.20 | 0.50 | 67 | 0 | 1 |
| validation | 0.02 | 0.50 | 68 | 15 | 0 |
| test | 0.20 | 0.50 | 79 | 1 | 3 |
| test | 0.02 | 0.50 | 80 | 21 | 2 |

상세 예측 결과는 `results/test/`와 `results/validation/`에서 확인할 수 있다.
