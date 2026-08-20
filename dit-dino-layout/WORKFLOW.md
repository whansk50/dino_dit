# Workflow and architecture

## 1. 구성

```text
dit-dino-layout/
├── src/dit_layout_bench/
│   ├── resources/               # default.yaml과 DINO 내부 bridge
│   ├── backends/                # 시작 시 detector 하나를 선택
│   ├── models/dit.py            # MPViT 비사용 clean-room DiT-base
│   ├── models/pyramid.py        # 공통 stride 4/8/16/32 pyramid + mask
│   ├── checkpoint.py            # hash/키/shape 검증 및 pos-embed 보간
│   ├── config.py                # strict YAML merge/validation
│   ├── data.py, spec.py         # PubLayNet I/O와 category 단일 정의
│   └── tracking.py              # 공통 MLflow run과 metric logger
│   ├── _vendor/dino/            # Apache-2.0 DINO 고정 소스와 LICENSE
├── train.py, evaluate.py, inference.py
└── tests/
```

상위 workspace의 `dit/`와 `DINO/`는 조사용일 뿐 runtime dependency가 아니다.
`get_backend()`가 프로세스 시작 시 `--detector` 값으로 모듈 하나만 import하므로
forward 내부에 detector 조건 분기가 없다.

## 2. huggingface conda 환경 설치

```bash
conda activate huggingface
cd /workspace/dino_dit/dit-dino-layout
python -m pip install -e .
```

현재 torch/CUDA와 ABI가 맞는 Detectron2 설치가 필요하다. DINO를 실행하기 전에
vendored multi-scale deformable attention을 같은 환경에서 빌드한다.

```bash
bash scripts/build_dino_ops.sh
python scripts/smoke_dino_op.py
```

## 3. PubLayNet 준비

데이터 다운로드 코드는 프로젝트에 포함하지 않는다. Hugging Face에서 사용자가
별도로 받은 PubLayNet을 다음 COCO layout으로 materialize한다.

```text
publaynet/
├── train/*.jpg
├── val/*.jpg
└── annotations/
    ├── train.json
    └── val.json
```

실행 시 train/val 모두를 검사하며 category ID는 `1: Text`, `2: Title`,
`3: List`, `4: Table`, `5: Figure`여야 한다. 다운로드 위치나 Hugging Face
dataset revision은 실험 메타데이터와 함께 사용자가 관리한다.

## 4. Pretrained 정책

학습은 `--pretrained`가 필수다. DiT의 IIT-CDIP self-supervised base/16
checkpoint를 로컬 파일로 전달한다. 두 detector 모두 동일 loader를 거친다.
loader는 절대 경로, SHA-256, loaded/missing/unexpected/shape-mismatch key를
보고하며 `blocks.*`가 빠지면 즉시 중단한다. 크기가 다른 absolute position
embedding은 class token을 분리한 뒤 bicubic 보간한다. Pyramid와 detector는
무작위 초기화된다.

## 5. 동일 I/O와 feature 흐름

두 backend의 공통 정책은 `src/dit_layout_bench/resources/default.yaml` 한 곳에서 읽는다.

```text
RGB → horizontal flip(train) → short edge 480..800 / long edge ≤1333
    → normalize(mean=.5, std=.5) → stride-32 padding + boolean mask
    → DiT taps(3,5,7,11) → P2/P3/P4/P5(stride 4/8/16/32)
```

학습 target은 COCO box/category를 사용한다. 추론 결과는 detector와 관계없이
원본 이미지 좌표의 `box_xyxy`, float `score`, PubLayNet `category_id`,
`image_id`를 반환한다. DINO 고유 random crop은 공정 비교를 위해 끈 상태다.

## 6. YAML 설정

우선순위는 `default.yaml → --config partial.yaml → --options → 전용 CLI flag`다.
알 수 없는 key나 기본값과 타입이 다른 값은 거부한다.

```yaml
training:
  epochs: 24
  detector_lr: 5.0e-5
  backbone_lr: 5.0e-6
dino:
  num_queries: 500
tracking:
  experiment_name: publaynet-ablation
```

```bash
python train.py --config configs/experiment.yaml \
  --detector dino --data-root /data/publaynet \
  --pretrained /weights/dit-base.pth \
  --options training.batch_size=4 run.amp=true
```

## 7. 학습, 평가, 추론

```bash
python train.py --detector cascade_rcnn --data-root /data/publaynet \
  --pretrained /weights/dit-base.pth --output-dir outputs/cascade

python train.py --detector dino --data-root /data/publaynet \
  --pretrained /weights/dit-base.pth --output-dir outputs/dino

python evaluate.py --detector dino --data-root /data/publaynet \
  --resume outputs/dino/checkpoint.pth

python inference.py --detector dino --resume outputs/dino/checkpoint.pth \
  --image page.jpg --json-output prediction.json
```

Backbone parameter는 기본 LR `1e-5`, pyramid/detector는 `1e-4`의 별도 AdamW
group에 들어간다. DINO는 epoch loop, Cascade는 annotation image 수를 기준으로
epoch를 iteration으로 변환한다.

## 8. MLflow

`tracking.enabled: true`가 기본이다. run마다 detector/action tag, 완전히 병합된
`effective-config.yaml`, train loss/LR, evaluation metric을 기록한다. DINO는
학습 callback, Cascade는 Detectron2 EventWriter를 사용하지만 같은 logging
계약을 따른다. 로컬 기본 저장소는 `sqlite:///mlflow.db`이며 원격 server URI도 YAML로
변경할 수 있다. 끄려면 `--options tracking.enabled=false`를 사용한다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 9. 검증

```bash
conda run -n huggingface python -m pytest tests
conda run -n huggingface python -m compileall src vendor/dino
```

실데이터 full run 전에는 작은 COCO subset으로 두 detector 각각 forward,
backward, checkpoint resume, original-coordinate prediction 및 COCO bbox AP를
확인한다. 결과 비교 시 git revision, effective YAML, checkpoint hash, seed,
precision, batch size, GPU, optimizer step 수를 함께 보존한다.

## 10. 라이선스 경계

MPViT 유래 `dit/object_detection/ditod` 코드는 import·복사하지 않았다. DINO는
Apache-2.0 원본 소스와 LICENSE를 `_vendor/dino/`에 보존한다. DiT clean-room
구현과 상세 출처는 `THIRD_PARTY_NOTICES.md`에 기록했다. 최종 배포 전 별도
법률 검토가 필요하다.
