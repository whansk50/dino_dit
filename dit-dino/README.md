# DiT Layout Bench: DINO / Cascade R-CNN

PubLayNet 문서 레이아웃 검출을 위해 하나의 clean-room DiT-base backbone과
P2/P3/P4/P5 feature pyramid를 공유하고, detector만 DINO 또는 Cascade R-CNN으로
교체해 비교하는 연구용 프로젝트다.

```text
RGB image + COCO target
          │
          ▼
clean-room DiT-base/16
  └─ transformer layer 3/5/7/11 feature
          │
          ▼
shared 256-channel P2/P3/P4/P5 (stride 4/8/16/32)
          ├─ DINO
          └─ Cascade R-CNN
```

두 backend는 같은 PubLayNet category, 입력 전처리, backbone checkpoint loader,
optimizer parameter group, prediction JSON 형식과 MLflow 기록 계약을 사용한다.
학습과 추론 시에는 한 backend만 lazy import한다.

## 지원 범위

| 항목 | DINO | Cascade R-CNN |
|---|---|---|
| 단일 GPU 학습·추론 | 지원 | 지원 |
| 로컬 multi-GPU DDP (`--devices`) | 지원 | 지원 |
| 외부 `torchrun` DDP | 지원 | 지원 |
| FP16 AMP | 지원 | 지원 |
| DiT activation checkpointing | 지원 | 지원 |
| fresh/resume subset validator | `validate_dino_training.py` | `validate_cascade_training.py` |
| 추가 runtime 의존성 | vendored deformable-attention CUDA op | PyTorch/CUDA 호환 Detectron2 |

DDP는 CUDA/NCCL을 사용한다. BF16은 vendored deformable-attention op 검증 범위에
포함되지 않아 현재 설정으로 노출하지 않는다. DiT attention은 CUDA FP16에서
PyTorch Flash SDPA backend를 강제한다.

## 설치

Python 3.10 이상과 PyTorch 2.3 이상이 필요하다. 공통 의존성과 테스트 의존성을
editable mode로 설치한다.

```bash
cd /path/to/dit-dino
python -m pip install -e ".[test]"
```

validator는 checkout의 `src/`를 직접 bootstrap하므로 editable 설치 여부와 관계없이
실행할 수 있다. 실제 학습과 추론 진입점은 위 설치를 권장한다.

DINO를 사용할 때는 현재 PyTorch/CUDA 환경에서 vendored multi-scale deformable
attention extension을 빌드한다.

```bash
bash scripts/build_dino_ops.sh
```

Cascade R-CNN을 사용할 때는 현재 PyTorch/CUDA ABI와 맞는 Detectron2를 별도로
설치해야 한다. Detectron2는 `requirements.txt`나 `pyproject.toml`에서 임의 버전으로
고정하지 않는다. Cascade만 실행한다면 DINO CUDA op 빌드는 필요하지 않다.

## PubLayNet 데이터

데이터 다운로드 코드는 포함하지 않는다. 사용자가 준비한 PubLayNet은 다음 COCO
layout을 따라야 한다.

```text
publaynet/
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg
└── annotations/
    ├── train.json
    └── val.json
```

학습 시작 시 두 split의 디렉터리와 annotation을 검사한다. category mapping은
아래 값과 정확히 일치해야 한다.

| category_id | name |
|---:|---|
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

중복 image ID, 존재하지 않는 image/category를 참조하는 annotation, 잘못된 COCO
root도 실행 전에 거부한다.

## DiT pretrained checkpoint

새 학습에는 `paths.pretrained` 또는 `--pretrained`로 self-supervised DiT-base/16
checkpoint 파일을 전달해야 한다. DINO와 Cascade 모두 동일한 loader를 사용한다.

loader는 다음을 수행한다.

- legacy UniLM checkpoint metadata를 PyTorch safe loader로 읽는다.
- prefix를 정규화하고 encoder key와 tensor shape를 검사한다.
- 입력 해상도가 달라지면 absolute position embedding을 bicubic 보간한다.
- checkpoint 절대 경로, SHA-256과 loaded/missing/unexpected/shape-mismatch 수를
  출력한다.
- 필수 encoder parameter가 누락되면 학습을 중단한다.

Pyramid와 detector head는 새로 초기화된다. `--resume`을 사용하면 full detector
checkpoint를 복원하므로 pretrained checkpoint는 다시 읽지 않는다.

## 설정 파일

실험용 전체 설정은 다음 두 파일에서 시작한다.

- `configs/dino_train.yaml`
- `configs/cascade_rcnn_train.yaml`

코드에 포함된 backend별 기본값은 다음 package resource에 있다.

- `src/dit_layout_bench/resources/dino.yaml`
- `src/dit_layout_bench/resources/cascade_rcnn.yaml`

설정 우선순위는 다음과 같다.

```text
backend 기본 YAML → --config YAML → --options KEY=VALUE → 전용 CLI flag
```

알 수 없는 key, 기본값과 타입이 다른 값, 선택한 `--detector`와 config의
`run.detector`가 다른 경우는 거부한다. 새 실험은 해당 backend config를 복사해
버전 관리하고, 일회성 변경에만 `--options`를 사용하는 것을 권장한다.

```bash
python train.py \
  --config configs/dino_train.yaml \
  --options training.epochs=24 dino.num_queries=500
```

주요 공통 section은 다음과 같다.

- `run`: detector, device, seed, process당 DataLoader worker 수, AMP
- `paths`: PubLayNet, output, checkpoint, pretrained 경로
- `input`: resize, normalization, horizontal flip
- `training`: global batch, epoch, LR, weight decay, warmup, 평가 주기
- `dit`: drop path, activation checkpointing, pyramid channel 수
- `tracking`: MLflow URI, experiment/run 이름, logging 주기
- `dino` 또는 `cascade_rcnn`: 선택한 detector 전용 설정

## 학습

config 파일의 경로를 실제 데이터와 checkpoint 위치에 맞게 수정한 뒤 실행한다.

```bash
# DINO 단일 GPU
python train.py --config configs/dino_train.yaml

# Cascade R-CNN 단일 GPU
python train.py --config configs/cascade_rcnn_train.yaml
```

config를 수정하지 않고 CLI로 필수 경로를 전달할 수도 있다.

```bash
python train.py \
  --detector dino \
  --data-root /data/publaynet \
  --output-dir outputs/dino-fpn256 \
  --weights-dir outputs/dino-fpn256/weights \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --batch-size 2 \
  --amp
```

### 로컬 multi-GPU DDP

GPU ID를 두 개 이상 지정하면 `train.py`가 선택한 GPU마다 process를 하나씩
생성한다. 두 backend 모두 같은 launcher를 사용한다.

```bash
python train.py --devices 0,1 --config configs/dino_train.yaml
python train.py --devices 2,3 --config configs/cascade_rcnn_train.yaml
```

`--devices`의 ID는 부모 process에서 PyTorch에 보이는 CUDA device 기준이다.
launcher는 선택된 device만 자식 process의 `CUDA_VISIBLE_DEVICES`에 노출하고
`LOCAL_RANK=0..N-1`로 재매핑한다. 따라서 `--devices 2,3`이나 기존
`CUDA_VISIBLE_DEVICES` 환경에서도 PyTorch와 Detectron2의 local-rank 계약이
일치한다.

`training.batch_size`와 `--batch-size`는 항상 global batch다. world size로 정확히
나누어지지 않으면 process를 생성하기 전에 거부한다. 예를 들어 global batch 6을
GPU 두 장에서 실행하면 GPU당 batch는 3이다. `run.num_workers`는 process당 값이다.

GPU 하나만 지정하면 DDP를 만들지 않고 해당 device에서 실행한다.

```bash
python train.py --devices 2 --config configs/dino_train.yaml
python train.py --devices 2 --config configs/cascade_rcnn_train.yaml
```

외부 launcher를 사용할 때는 `--devices`를 함께 전달하지 않는다.

```bash
torchrun --standalone --nproc-per-node=2 \
  train.py --config configs/cascade_rcnn_train.yaml
```

checkpoint, config, log와 MLflow run은 global rank 0만 기록한다. 평가 prediction과
loss 통계는 모든 rank에서 모은다. 동일한 pyramid 설정의 checkpoint는 단일 GPU와
DDP 사이에서 호환된다.

### 학습 재개

학습 재개 대상은 오직 `paths.weights_dir/recent.pth`다.

```bash
python train.py --config configs/dino_train.yaml --resume
python train.py --config configs/cascade_rcnn_train.yaml --resume
```

임의 checkpoint 경로를 `--resume`에 전달하거나 output 디렉터리에서 checkpoint를
자동 탐색하지 않는다. 설정된 평가 주기와 마지막 epoch의 validation이 성공하면
`recent.pth`를 갱신한다. resume 시 model, optimizer, LR scheduler, AMP 활성화 시
scaler와 완료된 epoch/iteration 상태를 복원한다.

현재 pyramid는 stride-16 DiT feature를 먼저 `768 → pyramid_channels`로 projection한
뒤 P2/P3/P4/P5로 변환한다. projection 이전에 768 channel로 pyramid를 만들던 legacy
detector checkpoint는 shape와 optimizer state가 달라 resume할 수 없다. 이 경우
동일한 DiT pretrained checkpoint에서 새 detector 학습을 시작해야 한다.

## PubLayNet subset runtime 검증

`smoke_*` 분기나 테스트 전용 동작을 본 학습 코드에 넣지 않는다. 별도 validator가
원본 PubLayNet 이미지의 symlink와 축소 COCO annotation을 임시 작업 디렉터리에 만든
뒤 실제 `train.py` 경로를 실행한다. 원본 dataset은 수정하지 않는다.

```bash
python scripts/validate_dino_training.py \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --devices 2

python scripts/validate_cascade_training.py \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --devices 2
```

`--devices 2`처럼 ID 하나를 전달하면 지정한 GPU의 single-process 경로를 검증한다.
`--devices 0,1`처럼 둘 이상을 전달하면 같은 validator가 DDP 경로를 검증한다.

두 validator는 fresh 학습과 resume 학습을 순서대로 실행하고 다음을 검사한다.

- 실제 GPU forward/backward 및 평가(2개 이상이면 DDP와 분산 평가)
- FP16 AMP와 DiT activation checkpointing
- validation 이후 `recent.pth` 생성
- optimizer, LR scheduler, AMP scaler 및 진행 상태 저장·복원
- MLflow run의 single/DDP tag와 world-size metadata(DDP에서는 rank 0만 기록)

DINO validator는 fused AdamW를 항상 검사하고, multi-GPU에서는 gradient bucket
view와 static-graph DDP도 검사한다.
기본 실행은 빠른 검증을 위해 DINO encoder/decoder layer 수를 줄인다. 전체 DINO
detector layer로 검증하려면 `--full-detector`를 추가한다. Cascade validator는
Cascade 구조를 유지하고 proposal sampling batch만 줄인다.

자동 생성된 작업 디렉터리는 성공하면 삭제되고 실패하면 분석을 위해 남는다.
`--work-dir`에는 비어 있는 디렉터리를 지정할 수 있으며, 자동 생성 디렉터리를
성공 후에도 유지하려면 `--keep-work-dir`를 사용한다.

## 추론

추론은 full detector checkpoint가 필요하므로 항상 `--resume`을 요구한다. 입력은
지원 이미지 파일 하나 또는 이미지가 직접 들어 있는 디렉터리다. 디렉터리는
재귀 탐색하지 않는다.

```bash
# JSON은 stdout, visualization은 OUTPUT_DIR/inference에 저장
python inference.py \
  --config configs/dino_train.yaml \
  --resume \
  --image page.jpg

# 폴더 추론, JSON/visualization 경로 지정
python inference.py \
  --config configs/cascade_rcnn_train.yaml \
  --resume \
  --image pages \
  --score-threshold 0.6 \
  --json-output outputs/cascade-predictions.json \
  --visualization-dir outputs/cascade-visualizations

# visualization 없이 JSON만 생성
python inference.py \
  --config configs/dino_train.yaml \
  --resume \
  --image pages \
  --json-output outputs/predictions.json \
  --no-visualize
```

지원 확장자는 BMP, JPEG/JPG, PNG, TIFF/TIF, WebP다. `--score-threshold`를 생략하면
선택한 detector YAML의 `score_threshold`를 사용한다. visualization은 기본적으로
활성화된다.

두 backend의 prediction JSON record 형식은 동일하다.

```json
{
  "image_id": "page",
  "box_xyxy": [72.5, 104.0, 530.25, 688.75],
  "score": 0.97,
  "category_id": 1
}
```

box는 원본 이미지 좌표의 XYXY 형식이고 `category_id`는 PubLayNet의 1-based ID다.

## MLflow

MLflow tracking은 기본적으로 활성화된다. 학습과 추론 run에 다음을 기록한다.

- detector, action, distributed tag
- 완전히 병합된 `effective-config.yaml`
- 학습 run의 global/per-GPU batch와 모든 run의 world size
- train loss/LR 및 COCO evaluation metric
- 추론 이미지와 prediction 수

기본 local backend store는 `sqlite:///mlflow.db`다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

다른 tracking server는 `tracking.tracking_uri`로 지정하고, 끄려면 다음 override를
사용한다.

```bash
python train.py \
  --config configs/dino_train.yaml \
  --options tracking.enabled=false
```

## 테스트

정적·계약·optimizer/model 테스트는 실제 PubLayNet이나 Detectron2 없이 실행할 수
있다.

```bash
python -m pytest -q
python -m compileall -q train.py inference.py src scripts tests
git diff --check
```

실제 GPU, CUDA extension, DDP, checkpoint resume까지 확인하려면 앞의 backend별
PubLayNet subset validator를 사용한다.

## 주요 구조

```text
dit-dino/
├── train.py                         # 학습 및 local DDP 진입점
├── inference.py                     # 단일/폴더 추론 및 visualization
├── configs/                         # 버전 관리하는 backend별 실험 YAML
├── scripts/
│   ├── build_dino_ops.sh            # vendored DINO CUDA op 빌드
│   ├── publaynet_subset.py           # validator 공통 subset 생성
│   ├── validation_runtime.py          # checkout src/와 child 환경 bootstrap
│   ├── validate_dino_training.py     # DINO actual-runtime 검증
│   └── validate_cascade_training.py  # Cascade actual-runtime 검증
├── src/dit_layout_bench/
│   ├── arguments.py                 # CLI와 config 생성
│   ├── config.py                    # strict YAML merge와 RunConfig
│   ├── launcher.py                  # local rank/GPU remapping과 spawn
│   ├── runtime.py                   # CUDA/process-group lifecycle
│   ├── backends/                    # DINO/Cascade adapter
│   ├── models/                      # clean-room DiT와 shared pyramid
│   ├── resources/                   # package 기본 YAML과 DINO bridge config
│   ├── data.py, spec.py             # PubLayNet 데이터/category 계약
│   ├── checkpoint.py                # pretrained/resume shape 검증
│   ├── prediction.py                # 공통 prediction/visualization
│   ├── tracking.py                  # rank-aware MLflow
│   └── _vendor/dino/                # vendored IDEA DINO + Apache-2.0 LICENSE
└── tests/
```

상위 workspace의 다른 DiT/DINO checkout은 runtime dependency가 아니다. DINO 소스는
`src/dit_layout_bench/_vendor/dino/`에 고정되어 있고, backbone/FPN은 MPViT wrapper를
import하거나 복사하지 않고 표준 PyTorch 연산으로 구현했다.

더 자세한 내부 구조와 실험 계약은 [WORKFLOW.md](WORKFLOW.md), 제3자 구성요소와
라이선스 경계는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), 남은 비교 실험은
[TODO.md](TODO.md)를 참고한다.
