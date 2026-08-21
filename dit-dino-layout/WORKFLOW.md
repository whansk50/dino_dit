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

`configs/dino_train.yaml`은 DINO 학습에 필요한 실행 경로와 실험 설정을
버전 관리한다. 새 실험은 이 파일을 복사해 수정한다. 일회성 비교 실험에만
`--options training.batch_size=4`와 같은 override를 사용한다.

```bash
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc-per-node=gpu \
  --numa-binding=node train.py --config configs/dino_train.yaml
```

## 7. 학습, 평가, 추론

```bash
python train.py --config configs/dino_train.yaml

# 보이는 GPU 수에 따라 single/DDP 자동 선택 (DINO backend)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=gpu \
  --numa-binding=node \
  train.py --config configs/dino_train.yaml

# 완료된 마지막 epoch부터 명시적으로 학습 재개
python train.py --config configs/dino_train.yaml \
  --resume outputs/dino-fpn256/checkpoint.pth

python evaluate.py --detector dino --data-root /data/publaynet \
  --resume outputs/dino-fpn256/checkpoint.pth

python inference.py --detector dino --resume outputs/dino-fpn256/checkpoint.pth \
  --image page.jpg --json-output prediction.json
```

`--resume`을 지정한 경우에만 모델, optimizer, LR scheduler 및 완료 epoch를
복원한다. resume에는 detector 전체 가중치가 있으므로 `paths.pretrained`는
생략할 수 있다. 출력 디렉터리에 기존 `checkpoint.pth`가 있더라도
`--resume`을 생략하면 pretrained 가중치에서 새 학습을 시작한다.

DDP는 `torchrun`의 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`를 감지해 자동 활성화된다.
일반 `python train.py ...` 실행은 기존 단일 GPU 동작을 유지한다. DINO의
`training.batch_size`와 `run.num_workers`는 GPU(process)당 값이므로 위 2-GPU
예제의 global batch는 `2 × training.batch_size`이고 DataLoader worker 수도
전체적으로 `2 × run.num_workers`가 된다. global batch가 달라지면 learning
rate와 수렴 특성이 달라질 수 있으므로 동일 조건 비교 시 이를 함께 기록한다.
checkpoint, config, log 및 MLflow run은 global rank 0만 기록하고, 평가 예측과
loss 통계는 모든 rank에서 모은다. 같은 pyramid 설정의 DDP checkpoint는 단일
GPU와 상호 호환된다.

현재 서버는 NVLink가 연결되어 있지 않고 GPU `0,1`과 `2,3`이 각각 같은 NUMA
측에 있다. 2-GPU 실행은 `CUDA_VISIBLE_DEVICES=0,1` 또는 `2,3`을 사용하고
`--numa-binding=node`를 유지한다. GPU 선택은 launcher/scheduler의 자원 할당
영역이므로 학습 코드가 임의로 바꾸지 않는다.

현재 pyramid는 네 stride-16 DiT tap을 먼저 768→256으로 projection한 뒤
P2/P3/P4/P5로 변환한다. 이전 768채널 pyramid와는 projection 및 transposed-conv
parameter shape가 다르고 DINO 입력 projection과 optimizer state도 달라서, 이전
detector checkpoint를 `--resume`할 수 없다. 오류 시 이를 명시적으로 안내한다.
DiT encoder 자체의 parameter shape는 바뀌지 않았으므로 동일한 self-supervised
`paths.pretrained`에서 새 output directory로 학습을 시작할 수 있다. AP-small
회귀 비교 조건은 [TODO.md](TODO.md)에 기록했다.

DataLoader는 pinned memory와 non-blocking H2D를 사용하고 train worker만 epoch
사이에 유지한다. `training.batch_size`, `run.num_workers`, prefetch는 rank당
값이다. Adam/AdamW fused optimizer는 CUDA에서만 켜고, DDP는 gradient bucket
view와 static graph를 사용한다. AMP dtype은 FP16으로 고정한다.

DINO의 linear warmup은 첫 epoch 안에서 `training.warmup_iters`만큼 적용된다.
평가는 `training.evaluate_every_epochs` 주기와 마지막 epoch에 실행한다.
`checkpoint.pth`는 안전한 재개를 위해 매 epoch 갱신하고, 번호가 붙은 보관본은
`training.checkpoint_every_epochs` 주기로 저장한다.

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
