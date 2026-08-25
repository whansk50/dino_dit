# Workflow and architecture

## 1. 구성

```text
dit-dino-layout/
├── train.py                      # 학습 공개 진입점
├── inference.py                  # 단일/폴더 추론 및 가시화 진입점
├── src/dit_layout_bench/
│   ├── resources/               # detector별 기본 YAML과 DINO 내부 bridge
│   ├── arguments.py            # train/inference 공통 인자와 config 생성
│   ├── backends/                # 시작 시 detector 하나를 선택
│   ├── models/dit.py            # MPViT 비사용 clean-room DiT-base
│   ├── models/pyramid.py        # 공통 stride 4/8/16/32 pyramid + mask
│   ├── checkpoint.py            # hash/키/shape 검증 및 pos-embed 보간
│   ├── config.py                # detector 선택, strict YAML merge, RunConfig
│   ├── settings_validation.py   # 공통 및 detector별 설정 값 검증
│   ├── launcher.py              # 로컬 rank/GPU 매핑과 process spawn
│   ├── runtime.py               # CUDA device와 process-group lifecycle
│   ├── data.py, spec.py         # PubLayNet I/O와 category 단일 정의
│   ├── tracking.py              # 공통 MLflow run과 metric logger
│   └── _vendor/dino/            # Apache-2.0 DINO 고정 소스와 LICENSE
└── tests/
```

상위 workspace의 `dit/`와 `DINO/`는 조사용일 뿐 runtime dependency가 아니다.
`get_backend()`가 프로세스 시작 시 `--detector` 값에 해당하는 백엔드만 import하고
`train`/`build_predictor` 연산을 반환하므로 forward 내부에 detector 조건 분기가 없다.

백엔드 선택은 일반 Python 분기로 `cascade_rcnn` 또는 `dino`를 직접 lazy import한다.
DINO adapter의 `importlib` 사용은 upstream `main.py`가 `engine`, `models`, `util`을
최상위 모듈명으로 import하는 구조를 충돌 없는 별칭으로 한 번 로드하는 데만 남겨
두었다. Backbone, PubLayNet dataset, optimizer group, evaluation logging 및 train-step
logging은 monkey-patch하지 않고 `DinoIntegration` callback으로 DINO `main()`에
명시적으로 전달한다.

## 2. 환경 설치

```bash
cd /workspace/dino_dit/dit-dino-layout
python -m pip install -e .
```

현재 torch/CUDA와 ABI가 맞는 Detectron2 설치가 필요하다. DINO를 실행하기 전에
vendored multi-scale deformable attention을 같은 환경에서 빌드한다.

```bash
bash scripts/build_dino_ops.sh
```

운영 학습 코드에 smoke 전용 분기를 추가하지 않고 실제 통합 경로를 확인하려면
PubLayNet 일부를 symlink로 구성하는 별도 validator를 실행한다. 원본 dataset은
수정하지 않으며 성공 시 임시 dataset, checkpoint, MLflow DB를 자동 제거한다.

```bash
python scripts/validate_dino_training.py \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --devices 0,1

python scripts/validate_cascade_training.py \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --devices 0,1
```

각 validator는 해당 backend를 두 번 학습한다. 첫 실행에서 FP16 AMP, DiT activation
checkpointing, DDP와 validation/checkpoint 저장을 확인하고, 두 번째 실행에서 학습
재개와 rank 0 전용 MLflow 기록을 검증한다. DINO validator는 static-graph DDP와
fused AdamW도 확인하며 detector 전체 layer까지 실행하려면 `--full-detector`를
추가한다.

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

두 backend는 동일한 공통 key 계약을 유지하지만 기본 YAML은 분리되어 있다.
`resources/dino.yaml`에는 `dino` 섹션만, `resources/cascade_rcnn.yaml`에는
`cascade_rcnn` 섹션만 존재한다. DINO DataLoader prefetch나 EMA처럼 다른
backend가 소비하지 않는 key도 상대 config에는 넣지 않는다.

```text
RGB → horizontal flip(train) → short edge 480..800 / long edge ≤1333
    → normalize(mean=.5, std=.5) → stride-32 padding + boolean mask
    → DiT taps(3,5,7,11) → P2/P3/P4/P5(stride 4/8/16/32)
```

학습 target은 COCO box/category를 사용한다. 추론 결과는 detector와 관계없이
원본 이미지 좌표의 `box_xyxy`, float `score`, PubLayNet `category_id`,
`image_id`를 반환한다. DINO 고유 random crop은 공정 비교를 위해 끈 상태다.

## 6. YAML 설정

우선순위는 `detector별 기본 YAML → --config → --options → 전용 CLI flag`다.
알 수 없는 key나 기본값과 타입이 다른 값은 거부한다. config의
`run.detector`와 `--detector`가 다르면 실행하지 않으므로 DINO 설정을 Cascade에
잘못 적용하거나 그 반대가 되는 일을 막는다.

`configs/dino_train.yaml`과 `configs/cascade_rcnn_train.yaml`이 각각의 실행 경로와
실험 설정을 버전 관리한다. 새 실험은 해당 detector 파일을 복사해 수정한다.
일회성 비교 실험에만 `--options training.batch_size=4`와 같은 override를 사용한다.

```bash
python train.py --devices 2 \
  --config configs/dino_train.yaml
```

## 7. 학습, 평가, 추론

```bash
python train.py --config configs/dino_train.yaml

python train.py --config configs/cascade_rcnn_train.yaml

# train.py가 선택한 GPU마다 PyTorch DDP process를 하나씩 실행
python train.py --devices 0,1 --config configs/dino_train.yaml
python train.py --devices 0,1 --config configs/cascade_rcnn_train.yaml

# 완료된 마지막 epoch부터 명시적으로 학습 재개
python train.py --config configs/dino_train.yaml --resume

python inference.py --detector dino \
  --weights-dir outputs/dino-fpn256/weights --resume \
  --image page.jpg --json-output prediction.json

# 폴더 안의 모든 지원 이미지 추론 및 박스 가시화
python inference.py --config configs/dino_train.yaml --resume \
  --image pages --json-output outputs/predictions.json \
  --visualization-dir outputs/visualizations

python inference.py --config configs/cascade_rcnn_train.yaml --resume \
  --image pages --json-output outputs/cascade-predictions.json
```

`--resume`을 지정한 경우에만 모델, optimizer, LR scheduler 및 완료 epoch를
`paths.weights_dir/recent.pth`에서 복원한다. 다른 디렉터리나 파일명은 탐색하지
않는다. resume에는 detector 전체 가중치가 있으므로 `paths.pretrained`는 무시한다.
`--resume`을 생략하면 pretrained 가중치에서 새 학습을 시작한다.

`--devices`에 GPU ID를 두 개 이상 지정하면 `train.py`가
`torch.multiprocessing.spawn`으로 process를 만들고, 각 process에서
`torch.distributed`와 `DistributedDataParallel`을 초기화한다. `--devices`를
생략하거나 하나만 지정하면 단일 process로 실행한다. ID는 PyTorch에 현재
보이는 CUDA device 기준이다. launcher는 선택된 device만 자식 process에 노출하고
표준 local rank `0..N-1`로 재매핑하므로 비연속 device 선택도 Detectron2 DDP와
호환된다. 두 backend 모두 `training.batch_size`는 global batch이고
`run.num_workers`는 GPU(process)당 값이다. global batch 6을 GPU 두 장에서
실행하면 GPU당 batch는 3이며, 전체 DataLoader worker 수는
`2 × run.num_workers`가 된다. global batch가 world size로 나누어떨어지지 않으면
process 생성 전에 실행을 거부한다.
checkpoint, config, log 및 MLflow run은 global rank 0만 기록하고, 평가 예측과
loss 통계는 모든 rank에서 모은다. 같은 pyramid 설정의 DDP checkpoint는 단일
GPU와 상호 호환된다.

현재 서버는 NVLink가 연결되어 있지 않고 GPU `0,1`과 `2,3`이 각각 같은 NUMA
측에 있다. 2-GPU 실행은 `--devices 0,1` 또는 `--devices 2,3`을 사용한다.
NUMA binding은 scheduler의 자원 할당 영역이며 학습 코드가 변경하지 않는다.

현재 pyramid는 네 stride-16 DiT tap을 먼저 768→256으로 projection한 뒤
P2/P3/P4/P5로 변환한다. 이전 768채널 pyramid와는 projection 및 transposed-conv
parameter shape가 다르고 DINO 입력 projection과 optimizer state도 달라서, 이전
detector checkpoint를 `--resume`할 수 없다. 오류 시 이를 명시적으로 안내한다.
DiT encoder 자체의 parameter shape는 바뀌지 않았으므로 동일한 self-supervised
`paths.pretrained`에서 새 output directory로 학습을 시작할 수 있다. AP-small
회귀 비교 조건은 [TODO.md](TODO.md)에 기록했다.

DataLoader는 pinned memory와 non-blocking H2D를 사용하고 train worker만 epoch
사이에 유지한다. `training.batch_size`는 global 값이고, `run.num_workers`와
prefetch는 rank당 값이다. Adam/AdamW fused optimizer는 CUDA에서만 켜고, DDP는
gradient bucket view와 static graph를 사용한다. AMP dtype은 FP16으로 고정한다.

DINO의 linear warmup은 첫 epoch 안에서 `training.warmup_iters`만큼 적용된다.
평가는 `training.evaluate_every_epochs` 주기와 마지막 epoch에 실행한다.
validation이 성공할 때마다 `paths.weights_dir/recent.pth`를 고정된 재개 대상으로
갱신한다. 마지막 epoch에도 validation을 실행하므로 이 파일이 최종 가중치가 된다.

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
