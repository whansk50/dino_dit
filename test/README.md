# DiT + DINO PubLayNet Benchmark

`../DiT + DINO PubLayNet Benchmark 계획.md`에서 정의한 실험을 실제로 돌리기 위한 작업 트리.
`../dit`(baseline, 원본)와 `../DINO`(experiment, 원본)에서 실제로 쓰는 부분만 복제하고,
그 위에 DiT -> DINO 연결 코드와 공통 평가/계측 스크립트를 얹어 두 실험을 동일 조건에서
학습·평가·비교한다. 자세한 설계 근거는 계획 문서를 참고.

## 구조

```text
test/
├── data/publaynet/       # 데이터 (커밋 안 함, data/README.md 참고)
├── weights/               # 사전학습 weight (커밋 안 함, weights/README.md 참고)
├── common/                # 프레임워크 비의존 - COCO json 채점/취합만
├── baseline_dit_cascade/  # DiT + Cascade R-CNN (detectron2)
└── dit_dino/              # DiT + DINO (순수 PyTorch)
```

## 0. 환경 설정

```bash
pip install -r requirements.txt

# detectron2 (소스 빌드)
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2/

# MSDeformAttn (DINO 쪽 CUDA 확장, GPU 필요)
cd dit_dino/models/dino/ops && python setup.py build install && cd -
```

`data/`, `weights/`를 각각의 README에 따라 채운다.

## 1. 스모크 테스트 (학습 전 필수)

```bash
cd dit_dino
python tools/check_dit_backbone.py --weights ../weights/dit-base-224-p16-500k-62d53a.pth

cd ../common
python make_subset.py subset --src ../data/publaynet/train.json --dst /tmp/train_subset.json --n 100
python make_subset.py check ../data/publaynet/val.json ../data/publaynet/val.json
```

## 2. Baseline A 재현

```bash
cd baseline_dit_cascade
python train_net.py --config-file publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus <N>
```

## 3. Experiment B 학습

```bash
cd dit_dino
bash scripts/train_dit_dino.sh
```

## 4. 평가 & 결과표

```bash
cd common
python coco_perclass_eval.py --gt ../data/publaynet/val.json \
    --dt ../baseline_dit_cascade/output/inference/coco_instances_results.json \
    --name "DiT + Cascade R-CNN" --out eval_baseline.json
python coco_perclass_eval.py --gt ../data/publaynet/val.json \
    --dt ../dit_dino/logs/dit_dino/coco_instances_results.json \
    --name "DiT + DINO" --out eval_dino.json

python ../baseline_dit_cascade/bench_speed_d2.py \
    --config-file ../baseline_dit_cascade/publaynet_configs/cascade/cascade_dit_base.yaml \
    --out speed_baseline.json
python ../dit_dino/bench_speed_dino.py \
    --config-file ../dit_dino/config/DINO/DINO_4scale_dit.py \
    --resume ../dit_dino/logs/dit_dino/checkpoint.pth --out speed_dino.json

python collect_results.py \
    --baseline-name "DiT + Cascade R-CNN" --baseline-eval eval_baseline.json --baseline-speed speed_baseline.json \
    --exp-name "DiT + DINO" --exp-eval eval_dino.json --exp-speed speed_dino.json \
    --out benchmark_table.md
```

## 5. Ablation (B에 경쟁력이 있을 때만)

`dit_dino/config/DINO/DINO_4scale_dit.py`의 `dit_pyramid_strides`(B2), `num_queries`(B3),
`data_aug_scales`/`data_aug_max_size`(B4)를 바꿔가며 반복한다.

## 출처 / 라이선스

개인 연구·비배포 기준으로는 별도 조치가 필요 없다. git에 올릴 계획이라면(특히 public repo)
`baseline_dit_cascade/LICENSE`(MIT, microsoft/unilm)와 `dit_dino/LICENSE`(Apache-2.0, DINO)를
유지하고, 복제한 파일의 원본 헤더를 지우지 않는다. 자세한 내용은 계획 문서의 "라이선스 메모"
절 참고.
