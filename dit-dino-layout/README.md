# DiT + DINO layout detection

PubLayNet 문서 레이아웃 검출을 위해 DiT-base를 공통 backbone으로 사용하고,
실행 시 detector를 `cascade_rcnn` 또는 `dino`로 선택하는 연구용 구현이다.

```text
RGB/COCO PubLayNet I/O
        ↓
clean-room DiT-base → shared P2/P3/P4/P5
                         ├─ Cascade R-CNN
                         └─ DINO
```

```bash
conda activate huggingface
cd dit-dino-layout
python -m pip install -e .
bash scripts/build_dino_ops.sh

python train.py --detector dino \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth

# DINO on every visible GPU (one process per GPU, NUMA-local CPU binding)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=gpu \
  --numa-binding=node \
  train.py --config configs/dino_train.yaml
```

모든 사용자 하이퍼파라미터는 버전 관리되는 `src/dit_layout_bench/resources/default.yaml`에 있다.
`--config`로 partial YAML을 병합하고 `--options section.key=value`로 한 번 더
덮어쓸 수 있다. 학습·평가·추론은 기본적으로 MLflow에 effective config와
metric을 기록한다.

AMP는 현재 FP16만 사용한다. DiT attention은 CUDA FP16에서 PyTorch Flash
SDPA backend를 강제하며, BF16은 vendored deformable-attention CUDA op 검증 전까지
노출하지 않는다.

이 폴더는 상위의 `dit/` 또는 `DINO/`를 import하거나 경로로 참조하지 않는다.
DINO의 Apache-2.0 코드는 `src/dit_layout_bench/_vendor/dino/`에 라이선스와 함께 고정했고, DiT
backbone/FPN은 MPViT 코드를 사용하지 않고 표준 PyTorch 연산으로 구현했다.

전체 설치, 데이터 계약, 실행 절차와 구조는 [WORKFLOW.md](WORKFLOW.md)를 참고한다.
