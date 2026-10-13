# 제출용 소스코드: YOLO11s + ROI + 공유 품질

최종 제출 구성은 **색상 표시 제거·인페인팅 → YOLO11s → OOF 후보로 학습한 좌우 반전 ROI 검증 → joint_cluster NMS → 공유 품질 좌표 보정**이다. 학습 완료 가중치, 원본·전처리 데이터, 재학습 코드, 테스트데이터 예측결과가 모두 포함되어 있다. 압축을 해제한 이 폴더만으로 실행할 수 있다.

## 제출 요구사항과 파일 위치

| 요구사항 | 위치 |
|---|---|
| 소스코드 및 환경 | `src/`, `requirements.txt` |
| 학습용 데이터 | `data/raw/`, `data/processed/`, `data/train.txt` |
| README | 이 파일 |
| 테스트데이터 예측결과 | `results/test/predictions.csv`, `results/test/predictions.json` |
| 학습 완료 모델 | `models/detector.pt`, `models/verifier.pt`, `models/shared_quality.pt` |

데이터는 제공된 BMP 500장과 공식 YOLO TXT 500개, 표시 제거 후 PNG 500장 및 동일 라벨로 구성된다. `data/manifest.csv`와 `train.txt`·`validation.txt`·`test.txt`가 분할을 고정한다. 탐지기 학습 360장, 검증 68장, 테스트 72장이다. 공유 품질 모델은 학습 360장 중 OOF 301장만 사용했다. 검증·테스트 영상은 공유 품질 학습과 정규화에 사용하지 않았다.

## 설치와 실행

Python 3.12에서 압축을 해제한 폴더로 이동하여 실행한다. CPU 추론이 기본이며, 실행 시 모델을 다운로드할 필요가 없다.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows에서는 .venv\Scripts\activate
pip install -r requirements.txt
python src/test_model.py
python src/verify.py
python src/run.py test
python src/run.py validation
```

원본 BMP 한 장 또는 폴더를 입력하는 방법:

```bash
python src/run.py predict --input data/raw/images --output results/custom_predictions.json
```

입력은 BMP/PNG/JPG/JPEG다. 전처리한 회색 PNG는 그대로 사용할 수 있다. 사용자 입력의 JSON 출력 경로는 현재 작업 폴더 기준이며, 나머지 데이터·모델 경로는 프로젝트 폴더 기준이다. `test`와 `validation` 명령은 해당 결과 폴더의 예측·지표·표시 이미지를 다시 생성한다.

## 최종 모델

- YOLO11s: 640×640 입력, COCO 사전학습 초기화 후 360장으로 학습.
- ROI 검증: 64×64 회색 ROI와 박스 마스크, 박스 기하 특징. 두 출력 p(후보 타당성), q(IoU 품질)를 함께 학습하며 원본/좌우 반전의 손실을 평균한다.
- joint_cluster: p×q로 박스를 고르고 NMS IoU 0.5를 적용한다. 클러스터 최대 탐지 점수에 선택 박스의 p를 곱한다.
- 공유 품질: 84→64→32→1 구조. 301장 OOF 후보에서 생성한 86,913개 변형 표본으로 고정 48 epoch 학습. 원 후보의 최대 변이 32 px 이하일 때 최대 81개 좌표 변형을 평가하고 q≥0.55 및 Δq≥0.10을 만족하면 좌표를 바꾼다. 점수·박스 수·순서·판정 경로를 유지한다.

최종 가중치와 설정은 `configs/model.json`에 고정되어 있다. 공유 품질은 하나의 학습된 가중치이며 fold 앙상블이 아니다.

## 예측결과 형식과 확인된 성능

`predictions.json`은 테스트 72장 모두를 포함한다. 각 영상의 `predictions`는 원본 픽셀 좌표 `[x1,y1,x2,y2,score]` 목록이며 `refinement`에 좌표 보정 내역을 기록한다. 좌표의 원점은 영상 왼쪽 위다. `predictions.csv`는 박스당 한 행으로 `image_id,x1,y1,x2,y2,score,class_id,route` 열을 갖는다. `class_id=0`은 이물질이다. 박스가 없는 영상의 존재는 JSON과 `image_summary.csv`에서 확인한다.

| 세트 | 점수 기준 | IoU 매칭 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| validation 68장 | 0.20 | 0.50 | 67 | 0 | 1 |
| validation 68장 | 0.02 | 0.50 | 68 | 15 | 0 |
| test 72장 | 0.20 | 0.50 | 79 | 1 | 3 |
| test 72장 | 0.02 | 0.50 | 80 | 21 | 2 |

저장 결과는 점수 0.001 이상 전체 후보를 포함하며, 위 평가는 0.20·0.02에서 계산한다. 점수 0.20 이상은 `alarm_hold`, 0.02 이상만 존재하면 `weak_review`, 그 외는 `unconfirmed_no_candidate`다. 점수는 보정된 불량 확률이 아니다. 테스트는 이전 실험에서 관찰한 기존 holdout이며 새로운 블라인드 평가가 아니다. 정상 제품의 자동 통과 성능은 평가하지 않았다.

## 전처리 및 재학습

```bash
python src/run.py preprocess
python src/train.py detector --device cpu
python src/train.py verifier
python src/train_shared_quality.py train
```

탐지기·ROI 재학습은 포함된 COCO 초기 가중치와 OOF 후보를 사용한다. `--device mps`로 Apple Silicon 탐지기 학습을 실행할 수 있다. 재학습 산출물은 `retraining/`에 저장하며 제출 가중치를 덮어쓰지 않는다. 재학습한 가중치를 추론에 적용하려면 별도 복사본에서 모델 파일과 `configs/model.json`의 해시를 함께 갱신한다.

탐지기의 그룹 제외 OOF 후보부터 재생성:

```bash
python src/train.py oof --device cpu
python src/train.py verifier --raw retraining/oof_train_raw.json
```

공유 품질의 학습 특징부터 재생성:

```bash
python src/train_shared_quality.py features --features retraining/shared_quality_features.npz
python src/train_shared_quality.py train --features retraining/shared_quality_features.npz --output retraining/shared_quality_rebuilt
```

`data/shared_quality_oof301.json`은 301장을 제외한 나머지 59장만으로 학습한 기존 소스 탐지기·ROI 모델의 OOF 예측이다. 공유 품질 특징 재생성은 이 고정 예측, 포함된 전처리 영상, 공식 TXT를 사용한다. 정답은 학습 목표 계산에만 쓰이며 추론 시에는 사용하지 않는다. `data/shared_quality_features.npz`는 실제 학습에 쓴 특징 캐시이고 `configs/shared_quality_train.json`에 대상 ID와 설정을 기록했다. 소스 59장 모델의 원 실험 자체를 재실행하는 도구는 이 제출 패키지에 포함하지 않았다. 제공된 OOF 예측부터 공유 품질 모델을 재학습할 수 있다.

같은 환경에서도 장치·라이브러리 차이에 따라 재학습 가중치나 소수점 예측값이 달라질 수 있다. 제출 결과 재현에는 포함된 최종 가중치를 사용한다. 같은 환경에서 공유 품질 재학습의 상태 텐서와 정규화 값 일치를 확인했다.

## 검증 및 출처

`src/verify.py`는 파일 SHA256, 공식 라벨, 분할, OOF 대상, 예측 범위 및 저장 지표를 검사한다. `PACKAGE_VERIFICATION.json`에 이번 제출 패키지의 실행 검증 결과를 기록했다. `CHECKSUMS.json`은 ZIP 생성 직전 파일 기준이다. 재학습·재추론으로 파일을 변경한 뒤에는 기존 해시 검사와 다를 수 있다.

외부 학습 이미지는 추가하지 않았다. COCO 사전학습 YOLO11s는 Ultralytics 자산이며 라이선스는 `ULTRALYTICS_LICENSE.txt`에 포함했다. 결과보고서와 발표자료는 별도 제출 항목이다.
