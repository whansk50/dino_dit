# DiT + DINO PubLayNet Benchmark 계획

## 1. 목적

기존 UniLM DiT의 Document Layout Detection 구조인

```text
DiT Backbone
→ Feature Pyramid
→ RPN
→ Cascade R-CNN
```

과, DINO/detrex를 기반으로 DiT를 backbone으로 적용한

```text
DiT Backbone
→ Multi-scale Feature Pyramid
→ DINO
→ Layout Detection
```

을 동일한 **PubLayNet** 데이터셋에서 학습·평가하여 성능과 추론 비용을 비교한다.

핵심 목적은 **DiT의 문서 이미지 사전학습 표현은 유지하면서 Cascade R-CNN을 DINO로 교체했을 때의 효과를 검증하는 것**이다.

---

## 2. 비교 대상

### Baseline A — DiT + Cascade R-CNN

UniLM/DiT 공식 Object Detection 구조를 사용한다.

```text
Input Image
    ↓
DiT-base / patch16
    ↓
Feature Pyramid
    ↓
RPN
    ↓
Cascade ROI Heads
    ↓
BBox + Class
```

- Backbone: DiT-base
- Pretrained weight: `dit-base-224-p16-500k-62d53a.pth`
- Detector: Cascade R-CNN
- Framework: Detectron2 / UniLM DiT
- Dataset: PubLayNet

이 모델의 결과를 기준 성능으로 사용한다.

---

### Experiment B — DiT + DINO

DINO 또는 detrex를 기반으로 하고 기존 backbone을 DiT로 교체한다.

```text
Input Image
    ↓
DiT-base / patch16
    ↓
Multi-scale Feature Pyramid
    ↓
DINO Encoder
    ↓
DINO Decoder
    ↓
BBox + Class
```

- Backbone: DiT-base
- Pretrained weight: Baseline과 동일
- Detector: DINO
- Framework: DINO / detrex
- Dataset: PubLayNet

DINO 이후 별도의 DiT classifier를 연결하는 구조가 아니라, **DiT가 DINO의 backbone 역할을 수행한다.**

---

# 3. PubLayNet Dataset

PubLayNet의 5개 layout class를 사용한다.

| ID | Class |
|---:|---|
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

데이터는 COCO Detection 형식을 그대로 사용한다.

```text
publaynet/
├── train/
│   ├── PMCxxxx_00001.jpg
│   ├── PMCxxxx_00002.jpg
│   └── ...
├── val/
│   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

DINO와 DiT 양쪽 모두 **동일한 train/validation annotation 및 이미지**를 사용해야 한다.

---

# 4. DiT Backbone

두 실험 모두 동일한 DiT backbone을 사용한다.

```text
DiT-base
Patch Size = 16
Hidden Dimension = 768
Transformer Layers = 12
Attention Heads = 12
```

사전학습 weight:

```text
dit-base-224-p16-500k-62d53a.pth
```

기존 Cascade R-CNN detection checkpoint 전체를 DINO에 사용하는 것이 아니라 **DiT backbone pretrained weight만 공유**한다.

---

# 5. DiT → DINO 연결

DiT patch16의 기본 출력은 single-scale feature이다.

예:

```text
Input
1600 × 1152

      ↓ patch16

DiT Feature
100 × 72 × 768
```

DINO는 multi-scale feature를 사용하므로 DiT 출력에서 feature pyramid를 생성한다.

권장 구조:

```text
                     DiT Feature
                       stride 16
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
      upsample          identity          downsample
         ↓                 ↓                 ↓
     stride 8          stride 16         stride 32
         │                                   │
         │                               downsample
         │                                   ↓
         │                               stride 64
         │
         └──────────┬────────┬───────────────┘
                    ↓        ↓
                 DINO Encoder
                    ↓
                 DINO Decoder
```

예를 들어 입력 크기가 `1600 × 1152`라면:

```text
P3 : stride 8
     200 × 144

P4 : stride 16
     100 × 72

P5 : stride 32
     50 × 36

P6 : stride 64
     25 × 18
```

형태의 feature를 생성한다.

각 feature의 channel은 DINO에서 사용할 dimension으로 projection한다.

예:

```text
768
 ↓
1×1 Conv
 ↓
256
```

---

# 6. DINO 설정 초기안

초기 benchmark에서는 DINO의 일반적인 설정을 최대한 유지한다.

예시:

```yaml
num_queries: 300

hidden_dim: 256

num_feature_levels: 4

encoder_layers: 6

decoder_layers: 6

nheads: 8
```

Feature:

```text
P3 : stride 8
P4 : stride 16
P5 : stride 32
P6 : stride 64
```

PubLayNet은 한 페이지의 객체 수가 COCO 자연 이미지보다 지나치게 많지 않으므로 우선 `300 queries`로 시작한다.

---

# 7. Input Size

초기 실험에서는 두 detector가 가능한 한 동일한 이미지 resize 정책을 사용하도록 한다.

후보:

```text
short edge / long edge 제한 방식
```

또는 현재 DiT 환경과 맞추어:

```text
1280급
1600 × 1152급
```

을 사용한다.

단, 반드시 다음 조건을 맞춘다.

```text
Baseline A image resolution
≈
Experiment B image resolution
```

DINO만 더 높은 해상도를 사용하면 detector 자체의 성능 비교가 어려워진다.

---

# 8. Training 조건 통제

공정한 비교를 위해 최대한 동일하게 유지한다.

### 동일 조건

```text
Dataset
Train/Validation split
DiT pretrained weight
Input resolution
Augmentation
Class 정의
Evaluation data
Hardware
Mixed precision 여부
```

### Detector 특성상 달라질 수 있는 조건

```text
Optimizer
Learning rate
Training iteration
Warmup
Loss
LR scheduler
```

DINO와 Cascade R-CNN의 최적화 특성이 다르므로 optimizer와 LR까지 무조건 동일하게 맞출 필요는 없다.

대신 각각의 detector에서 합리적인 설정을 사용하고 그 값을 기록한다.

---

# 9. Learning Rate

DINO에서는 새로 초기화한 detector와 pretrained DiT backbone의 LR을 분리한다.

초기값 예:

```text
DINO:
1e-4

DiT backbone:
1e-5
```

즉:

```text
backbone_lr
=
detector_lr × 0.1
```

정도로 시작한다.

목적은 pretrained DiT 표현이 초기에 급격하게 망가지는 것을 방지하는 것이다.

---

# 10. 평가 지표

COCO evaluation metric을 기본으로 한다.

반드시 기록:

```text
AP
AP50
AP75
APsmall
APmedium
APlarge
```

특히 Document Layout Detection에서는 `AP75`를 중요하게 본다.

`AP50`이 높고 `AP75`가 낮다면:

```text
객체 자체는 찾지만
BBox 경계가 부정확
```

할 가능성이 있기 때문이다.

---

## 11. Class별 평가

전체 AP만 비교하지 않고 PubLayNet 5개 class별 AP를 비교한다.

| Class | Cascade R-CNN | DINO |
|---|---:|---:|
| Text | | |
| Title | | |
| List | | |
| Table | | |
| Figure | | |

특히 확인할 항목:

```text
Text ↔ Title confusion

Text ↔ List confusion

Table localization

Figure localization
```

---

# 12. 성능 외 평가

서비스 적용 가능성을 판단하려면 AP 외에도 다음을 측정한다.

### 추론 속도

동일 GPU에서:

```text
ms / page

pages / sec
```

측정.

가능하면 다음을 분리한다.

```text
preprocessing
model inference
postprocessing
total latency
```

---

### GPU Memory

측정 항목:

```text
Training peak VRAM
Inference peak VRAM
```

특히:

```text
DiT
+
Deformable Transformer Encoder
+
DINO Decoder
```

조합의 VRAM 사용량을 기존 Cascade R-CNN과 비교한다.

---

### Parameter Count

```text
Backbone params
Detector params
Total params
```

을 기록한다.

---

# 13. Benchmark 결과표

최종적으로 다음 표를 작성한다.

| Model | AP | AP50 | AP75 | APs | APm | APl | FPS | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiT + Cascade R-CNN | | | | | | | | |
| DiT + DINO | | | | | | | | |

Class별:

| Model | Text | Title | List | Table | Figure |
|---|---:|---:|---:|---:|---:|
| Cascade R-CNN | | | | | |
| DINO | | | | | |

---

# 14. 추가 Ablation

기본 성능 확인 후 다음 순서로 실험한다.

### Experiment B1

```text
DiT
+
DINO
+
P3/P4/P5/P6
```

기본 모델.

---

### Experiment B2 — Feature level

```text
P4/P5/P6
```

vs

```text
P3/P4/P5/P6
```

비교.

목적:

```text
Title
List
작은 Text region
```

등에서 stride-8 feature의 효과 확인.

---

### Experiment B3 — Query 수

```text
100
300
500
```

비교.

PubLayNet에서 실제 필요한 query 수를 확인한다.

---

### Experiment B4 — Input Resolution

예:

```text
1280급
vs
1600 × 1152급
```

비교.

특히 small-object AP와 추론시간의 trade-off를 확인한다.

---

# 15. 권장 실험 순서

처음부터 여러 설정을 변경하지 않는다.

### Step 1

기존 UniLM DiT PubLayNet baseline 재현.

```text
DiT
+
Cascade R-CNN
```

이를 기준 결과로 고정한다.

---

### Step 2

가장 단순한:

```text
DiT
+
4-scale pyramid
+
DINO
```

를 구현한다.

---

### Step 3

동일 validation set에서:

```text
AP
AP50
AP75
class AP
속도
VRAM
```

을 비교한다.

---

### Step 4

DINO가 baseline보다 경쟁력이 있을 경우에만:

```text
resolution
feature levels
query 수
learning rate
```

등을 튜닝한다.

---

# 16. 핵심 Benchmark 구조

최종적으로 검증하고자 하는 것은 아래 한 가지다.

```text
                동일한 PubLayNet
                       │
             동일한 DiT pretrained
                       │
             ┌─────────┴─────────┐
             │                   │
             ↓                   ↓
      Cascade R-CNN             DINO
             │                   │
             ↓                   ↓
        Detection            Detection
             │                   │
             └─────────┬─────────┘
                       ↓
                 COCO Evaluation
```

즉 **DiT 자체의 성능을 비교하는 것이 아니라 동일한 DiT backbone에서 Cascade R-CNN과 DINO 중 어떤 detector가 Document Layout Detection에 더 적합한지 비교하는 실험**으로 설계한다.

## 17. 1차 성공 기준

DINO 적용의 의미가 있다고 판단할 기준은 단순히 AP 하나만 높아지는 것이 아니다.

우선순위는:

1. `AP` 및 `AP75`가 Cascade R-CNN 이상
2. Table/Figure localization 개선
3. Title/Text/List 성능 유지 또는 개선
4. inference latency가 서비스 허용 범위
5. VRAM 증가가 감당 가능한 수준

정도로 둔다.

특히 `AP50`만 상승하고 `AP75`가 감소한다면 DiT+DINO가 기존 모델보다 명확하게 우수하다고 판단하지 않는다.