# DiT + DINO 문서 레이아웃 검출 구현 명세

## 1. 목적

문서 이미지에 특화된 사전학습 모델인 DiT를 공통 backbone으로 사용하고, detector를 실행 옵션으로 전환할 수 있는 문서 레이아웃 검출 시스템을 구현한다.

초기 구현에서는 다음 두 구성을 지원한다.

```text
DiT backbone → Feature Pyramid → Cascade R-CNN
DiT backbone → Feature Pyramid → DINO
```

DiT와 DINO 중 하나를 선택하는 구조가 아니다. DiT는 두 모델의 공통 feature extractor이고, Cascade R-CNN과 DINO가 교체 가능한 detector 역할을 맡는다.

주요 연구 목적은 동일한 데이터, DiT 사전학습 가중치, 입력 정책 아래에서 detector 변경에 따른 정확도와 계산 비용 차이를 비교하는 것이다.

## 2. 초기 구현 범위

### 포함

- DiT-base patch16 backbone
- PubLayNet 5개 클래스 문서 레이아웃 검출
- Cascade R-CNN detector
- DINO detector
- 실행 flag를 통한 detector 선택
- DiT 사전학습 checkpoint 로딩
- 공통 데이터 및 평가 규격
- 학습, 평가, 추론 진입점
- feature shape 및 checkpoint 로딩 검증
- 구현 완료 후 workflow 문서 작성
- MPViT 유래 코드 제거 및 독립 구현

### 초기 범위에서 제외

- DiT-large
- instance segmentation mask 출력
- DINO의 COCO detector checkpoint를 이용한 초기화
- 분산 학습 환경별 최적화
- 전체 하이퍼파라미터 탐색

## 3. 모델 구성

### 3.1 공통 DiT backbone

초기 모델은 다음 규격을 사용한다.

| 항목 | 값 |
|---|---|
| 모델 | DiT-base |
| Patch size | 16 × 16 |
| Hidden dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| 초기화 | IIT-CDIP self-supervised pretrained checkpoint |

기본 checkpoint:

```text
dit-base-224-p16-500k-62d53a.pth
```

두 detector 모두 동일한 DiT backbone checkpoint로 초기화한다. PubLayNet으로 이미 fine-tuning된 전체 detection checkpoint는 공정 비교용 초기화에 사용하지 않는다.

### 3.2 공통 feature pyramid

DiT patch16의 중간 layer feature를 받아 detector가 사용할 multi-scale feature를 생성한다.

초기 목표 stride는 다음과 같다.

```text
P2: stride 4
P3: stride 8
P4: stride 16
P5: stride 32
```

두 detector는 가능한 한 동일한 pyramid feature를 입력으로 사용한다. detector 구현상 추가 level이 필요하면 공통 pyramid의 마지막 level에서 생성하고 해당 차이를 문서에 기록한다.

각 level은 detector 내부 projection을 통해 필요한 channel dimension으로 변환한다. DINO의 기본 hidden dimension은 256으로 한다.

feature pyramid 구현에는 다음 동작이 포함되어야 한다.

- 입력 크기가 patch size 또는 최대 stride의 배수가 아닐 때 안전한 padding
- padding 영역을 나타내는 boolean mask 생성 및 전달
- 각 feature level의 실제 stride와 선언된 stride 일치
- 가변 해상도 및 직사각형 문서 이미지 지원
- backbone pretrained patch embedding shape 유지

## 4. Detector 전환 방식

사용자 인터페이스에서는 detector를 명시적으로 선택한다.

```bash
python train.py --detector cascade_rcnn ...
python train.py --detector dino ...
```

설정 값은 다음 두 값 중 하나만 허용한다.

```text
cascade_rcnn
dino
```

실행 중 매 forward마다 조건 분기하지 않고, 프로그램 시작 시 선택한 backend를 한 번 생성한다.

권장 모듈 경계는 다음과 같다.

```text
common/
  dataset_spec.py       # 클래스 및 category mapping
  transforms.py         # 공통 전처리 정책
  prediction.py         # 공통 추론 출력 형식
  evaluator.py          # 공통 COCO 평가 진입점

models/
  dit_backbone.py       # DiT backbone
  feature_pyramid.py    # 공통 multi-scale adapter
  cascade_backend.py    # Detectron2 Cascade R-CNN 연결
  dino_backend.py       # DINO 연결

train.py
evaluate.py
inference.py
```

실제 저장소 구조를 크게 훼손하지 않는 범위에서 위 책임 분리를 적용한다.

## 5. 데이터셋

초기 target dataset은 PubLayNet으로 고정한다.

클래스는 다음 5개다.

| Category ID | Class |
|---:|---|
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

COCO detection annotation 형식을 사용한다.

```text
publaynet/
├── train/
├── val/
└── annotations/
    ├── train.json
    └── val.json
```

두 detector는 반드시 동일한 이미지와 annotation split을 사용한다.

내부 학습 label을 0-based로 변환할 수 있으나, 변환 규칙은 한 곳에서 관리한다. 평가 및 최종 추론 출력에서는 원래 PubLayNet category ID로 복원한다.

## 6. 공통 I/O 규격

### 6.1 입력

두 detector에 다음 조건을 동일하게 적용한다.

- RGB 이미지
- 동일 train/validation split
- 동일 resize 후보 및 최대 long edge
- 동일 random horizontal flip
- 동일 crop 사용 여부
- 동일 normalization
- 동일 padding 규칙
- 동일 augmentation random seed 정책

초기 normalization은 DiT 사전학습 및 기존 DiT detection 설정에 맞춰 다음 값을 사용한다.

```text
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

이미지를 0~255 범위에서 처리하는 backend에서는 위 값을 각각 127.5로 환산한다. 두 표현이 실제로 동일하게 동작하는지 단위 테스트로 확인한다.

초기 resize 후보는 기존 DiT와 DINO가 공통으로 사용하는 다음 short-edge 범위를 기준으로 한다.

```text
480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800
```

long-edge 제한, crop 활성화 여부 및 validation resize는 config에 명시한다. 초기 공정 비교에서는 DINO에만 존재하는 추가 random crop을 비활성화한다.

### 6.2 정답 형식

공통 데이터 계층에서는 최소한 다음 값을 제공한다.

```text
image
image_id
original_size
processed_size
boxes
labels
```

box 좌표 변환은 backend adapter 내부에 한정한다.

### 6.3 추론 출력

두 backend의 외부 출력은 다음 형식으로 통일한다.

```text
boxes:  원본 이미지 좌표계의 xyxy
scores: float confidence
labels: PubLayNet category ID
image_id
```

NMS 적용 여부와 confidence threshold는 backend별 차이를 숨기지 않고 설정 및 결과 문서에 기록한다.

## 7. Pretrained checkpoint 정책

공정 비교의 기본 정책은 다음과 같다.

```text
DiT backbone: 동일한 self-supervised checkpoint로 초기화
Feature pyramid: 무작위 초기화
Cascade R-CNN detector: 무작위 초기화
DINO detector: 무작위 초기화
```

DINO의 COCO pretrained detector checkpoint를 사용하는 실험은 별도 ablation으로 취급한다.

checkpoint loader는 다음 항목을 출력하고 검증해야 한다.

- checkpoint 경로 및 hash
- 로드된 key 수
- missing keys
- unexpected keys
- shape mismatch keys
- transformer block 로드 여부
- 새로 초기화된 모듈 목록

`blocks.*` 또는 이에 해당하는 DiT transformer parameter가 로드되지 않으면 즉시 오류로 처리한다. Absolute position embedding의 크기가 다르면 bicubic interpolation을 적용하되, class token 등 extra token은 분리하여 보존한다.

## 8. DINO 초기 설정

초기 DINO 설정은 공식 4-scale 구성을 기준으로 하되 PubLayNet에 맞춰 조정한다.

```text
hidden_dim = 256
num_feature_levels = 4
enc_layers = 6
dec_layers = 6
nheads = 8
num_queries = 300
num_select = 300
use_dn = true
```

class 수와 labelbook 크기는 PubLayNet category mapping에 맞춘다. 배경 클래스를 실제 foreground class 수에 포함할지 여부는 DINO loss 구현을 확인해 한 곳에서 정의하며, magic number로 `5` 또는 `6`을 여러 파일에 중복하지 않는다.

초기 learning rate 후보는 다음과 같다.

```text
detector lr = 1e-4
backbone lr = 1e-5
```

backbone과 detector parameter group이 정확히 분리되었는지 parameter name과 개수를 시작 시 출력한다.

## 9. Cascade R-CNN 초기 설정

기존 UniLM DiT PubLayNet Cascade R-CNN 설정을 기준으로 한다.

다만 다음 부분은 공통 실험 조건에 맞게 조정한다.

- 공통 입력 normalization
- 공통 resize 및 augmentation
- 공통 DiT backbone checkpoint
- 공통 feature pyramid 규격
- 동일 PubLayNet category mapping
- 동일 평가 데이터와 evaluator

기존 PubLayNet fine-tuned Cascade R-CNN checkpoint는 기준 재현용 평가에는 사용할 수 있지만, DINO와의 공정 학습 비교 초기화에는 사용하지 않는다.

## 10. 학습 조건과 비교 원칙

다음 조건은 두 detector에서 동일하게 유지한다.

- 데이터셋 및 split
- DiT backbone 구조와 초기 checkpoint
- 입력 해상도 정책
- normalization
- augmentation
- class 정의
- 평가 데이터
- hardware
- mixed precision 사용 여부
- random seed 및 반복 실험 횟수

다음 조건은 detector 특성에 맞춰 다르게 설정할 수 있다.

- optimizer
- detector learning rate
- scheduler 및 warmup
- epoch 또는 iteration 수
- detector 고유 loss
- gradient clipping

서로 다른 값은 결과 비교 문서에 모두 기록한다. 단순히 epoch 수만 맞추지 않고 총 학습 이미지 수와 optimizer step 수도 함께 기록한다.

## 11. 평가

공통 COCO evaluator를 사용해 다음 지표를 기록한다.

- AP
- AP50
- AP75
- APsmall
- APmedium
- APlarge
- class별 AP

class별 AP는 다음 순서로 출력한다.

```text
Text, Title, List, Table, Figure
```

정확도 외에 다음 항목도 동일한 환경에서 측정한다.

- preprocessing latency
- model inference latency
- postprocessing latency
- total latency
- pages per second
- training peak VRAM
- inference peak VRAM
- backbone parameter 수
- detector parameter 수
- 전체 parameter 수

latency 측정 시 warm-up 횟수, 측정 반복 수, batch size, GPU 모델, precision을 기록한다.

## 12. 라이선스 및 코드 출처 정책

현재 `dit/object_detection` 일부 파일은 MPViT에서 유래했음을 명시하고 있다. MPViT는 GPLv3 또는 commercial license 조건을 가지므로 public repository 배포를 고려할 때 해당 구현을 그대로 사용하지 않는다.

재구현 원칙은 다음과 같다.

- MPViT source를 복사한 뒤 이름이나 표현만 변경하지 않는다.
- 필요한 동작을 입력, 출력, stride, shape 중심의 기능 명세로 분리한다.
- PyTorch, torchvision, Detectron2 등 사용 가능한 dependency API로 독립 구현한다.
- 교체 전후 feature shape 및 기능 동등성을 테스트한다.
- 새 구현에는 실제 참고한 프로젝트와 라이선스를 정확히 표기한다.
- MPViT header가 존재하는 파일뿐 아니라 관련 import와 파생 구현도 함께 감사한다.
- repository root에 최종 프로젝트 LICENSE와 THIRD_PARTY_NOTICES를 둔다.
- DINO의 Apache-2.0 LICENSE 및 NOTICE 의무를 유지한다.
- UniLM/DiT의 MIT 저작권 고지를 유지한다.

이 작업은 코드 출처 및 배포 위험을 줄이기 위한 기술적 조치다. 최종적인 법적 적합성은 별도 법률 검토 대상으로 한다.

## 13. 구현 순서

### Phase 1: 기반 정리

1. 현재 `dit/`와 `DINO/`의 실행 경로 및 dependency 조사
2. 사용 코드의 출처 및 라이선스 inventory 작성
3. PubLayNet category 및 공통 prediction schema 정의
4. 공통 resize, normalization 및 augmentation 규격 구현

### Phase 2: DiT 공통화

1. GPL 의존 없이 DiT backbone wrapper 독립 구현
2. pretrained checkpoint loader 구현
3. multi-scale feature pyramid 구현
4. 가변 크기 입력과 mask propagation 구현
5. feature shape, stride 및 weight loading 단위 테스트

### Phase 3: DINO 연결

1. DINO backbone factory에 DiT adapter 등록
2. channel projection 및 positional encoding 연결
3. PubLayNet dataset 및 class mapping 연결
4. 단일 batch forward/backward smoke test
5. 소규모 subset overfit test

### Phase 4: Cascade R-CNN 연결

1. 기존 DiT detector의 MPViT 유래 wrapper 교체
2. 공통 feature pyramid 연결
3. 공통 데이터 및 평가 규격 연결
4. 단일 batch forward/backward smoke test
5. 소규모 subset overfit test

### Phase 5: 통합 실행 경로

1. `--detector cascade_rcnn|dino` flag 구현
2. 공통 train/evaluate/inference interface 구현
3. config validation 및 명확한 오류 메시지 추가
4. checkpoint 저장 및 resume 동작 검증

### Phase 6: 문서 및 benchmark

1. 전체 PubLayNet 학습 및 평가 명령 정리
2. 성능, 속도, VRAM 및 parameter 결과 기록
3. `WORKFLOW.md` 작성
4. 라이선스와 third-party notice 최종 점검

## 14. 테스트 및 완료 기준

다음 조건을 모두 만족해야 초기 구현을 완료한 것으로 본다.

### 정적 및 단위 검증

- 두 detector flag가 유효한 backend를 생성한다.
- 잘못된 detector 값에 명확한 오류가 발생한다.
- 다양한 직사각형 입력에서 pyramid shape이 예상 stride와 일치한다.
- 모든 feature mask shape이 해당 feature와 일치한다.
- padding 영역이 attention 또는 loss에서 제외된다.
- checkpoint의 DiT transformer block이 정상 로드된다.
- category ID 변환 round-trip이 보존된다.
- 공통 normalization이 두 backend에서 수치적으로 동일하다.

### 실행 검증

- CPU 또는 가능한 환경에서 import test가 통과한다.
- 지원 GPU 환경에서 두 모델의 single-batch forward가 통과한다.
- 두 모델의 backward와 optimizer step이 통과한다.
- PubLayNet 소규모 subset에서 최소 1 epoch 학습이 완료된다.
- 동일 validation subset에서 COCO 평가 결과가 생성된다.
- 저장된 checkpoint로 resume 및 inference가 가능하다.

### 품질 검증

- subset overfit test에서 loss가 감소한다.
- 예측 box가 원본 이미지 좌표로 올바르게 복원된다.
- NaN 또는 Inf loss를 조기에 탐지한다.
- 학습 로그에 config, seed, dependency version, checkpoint 정보가 남는다.
- repository에 MPViT 유래 GPL 코드가 잔존하지 않는지 감사 결과가 남는다.

## 15. 최종 문서 산출물

구현 후 `WORKFLOW.md`에 다음 내용을 작성한다.

- 최종 directory 구조
- dependency 및 설치 방법
- CUDA extension 빌드 방법
- PubLayNet 준비 방법
- pretrained checkpoint 준비 및 검증 방법
- detector별 학습 명령
- 평가 및 추론 명령
- checkpoint resume 방법
- 공통 I/O와 내부 feature 흐름
- config 주요 항목
- benchmark 측정 방법
- 알려진 제한사항
- 사용 코드와 라이선스 출처

## 16. 후속 실험

초기 구현 완료 후 다음 순서로 확장한다.

1. DINO feature level `4/8/16/32`와 `8/16/32/64` 비교
2. query 수 100, 300, 900 비교
3. backbone learning rate 및 freeze 구간 비교
4. COCO pretrained DINO detector 초기화 실험
5. DiT-base와 DiT-large 비교

## 17. 구현 시 기본 결정 요약

별도 변경 요청이 없다면 다음 결정을 기준으로 구현한다.

```text
Dataset             = PubLayNet
Backbone            = DiT-base patch16
Backbone checkpoint = IIT-CDIP self-supervised pretrained
Detector switch     = --detector cascade_rcnn|dino
Common pyramid      = stride 4/8/16/32
Normalization       = mean/std 0.5
DINO hidden dim     = 256
DINO queries        = 300
Evaluation          = COCO bbox + class별 AP
License policy      = MPViT 유래 구현 제거 후 독립 재구현
```
