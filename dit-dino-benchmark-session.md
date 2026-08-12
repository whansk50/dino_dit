# DiT + DINO PubLayNet 벤치마크 — 작업 세션 요약

**날짜:** 2026-08-12
**요청:** `DiT + DINO PubLayNet Benchmark 계획.md`를 보고, `dit/`와 `DINO/` 두 원본 저장소에서 실제로 쓰는 부분만 복제해 `test/`에 두 실험(Baseline A: DiT+Cascade R-CNN, Experiment B: DiT+DINO)을 동일 조건에서 학습·평가·비교할 수 있게 구성.

---

## 1. 배경

기존 저장소에는 두 트리가 나란히 있을 뿐이었다.

- `dit/` — UniLM DiT. detectron2 기반, PubLayNet Cascade R-CNN 학습/평가가 이미 완결됨
- `DINO/` — IDEA DINO. 순수 PyTorch, COCO 전용, 백본은 ResNet/Swin/ConvNeXt만 지원

프레임워크·데이터 로더·평가 경로가 전부 달라 "동일 조건 비교"가 불가능한 상태였다. 목표는 두 폴더에서 실제로 쓰는 부분만 `test/`로 복제하고, 그 위에 DiT → DINO 연결 코드와 공통 평가·계측 스크립트를 얹어 하나의 트리에서 두 실험을 공정하게 비교하는 것.

## 2. 계획 수립 — 핵심 조사 결과

Plan 모드에서 두 코드베이스를 읽고 아래를 확인한 뒤 설계에 반영했다.

| 조사 항목 | 결론 |
|---|---|
| DiT 백본의 detectron2 의존성 | `beit.py`는 `torch`+`timm`만 사용 — detectron2에 묶인 건 래퍼(`backbone.py`)뿐. `beit.py`는 무수정 복제 가능 |
| DINO 4-scale 피라미드 만들기 | `beit.py`의 `patch_size==8` 분기 연산(2x ConvTranspose / Identity / MaxPool2 / MaxPool4)을 stride-16 tap에 그대로 적용하면 stride 8/16/32/64가 나옴 — 새로 발명할 것 없음 |
| 두 프레임워크의 augmentation | baseline(`MIN_SIZE_TRAIN 480~800`, crop 384~600)과 DINO(`coco_transformer.py` 480~800, crop 384~600)이 **이미 동일** — 해상도를 억지로 맞출 필요 없음 |
| 정규화 | DiT는 mean/std 0.5, DINO는 ImageNet 통계 — DiT 표현을 살리려면 DINO 쪽을 0.5로 맞춰야 함 |
| 라벨 | PubLayNet category_id 1~5을 그대로 쓰면 됨 (COCO가 91을 쓰는 것과 같은 이유), remap 불필요 |
| 패딩 | DINO는 배치 내 최대 크기로만 패딩 — stride-64 레벨을 쓰려면 백본 어댑터 안에서 64의 배수로 별도 패딩 필요 |
| 계측 도구 | DINO엔 params/GFLOPs/FPS 도구가 있지만 per-class AP가 없고, detectron2는 반대 — 공통 스크립트를 새로 작성해야 함 |

## 3. 라이선스 논쟁 — 가장 길었던 곁가지

### 1차 시도 (과했음)
`ditod/backbone.py`, `ditod/__init__.py`에서 "Dual License(GPL3.0 & Commercial)" 문구를 보고, `baseline_dit_cascade/`(GPL 구역)와 `dit_dino/`(Apache 구역)를 완전히 분리하는 구조(`LICENSE`/`NOTICE` 파일 세트, 구역 간 import 금지, grep 기반 분리 검증 등)를 계획에 넣었다.

### 사용자 피드백
"unilm 아래에 있어서 MIT로 알고 써왔는데, GPL이라는 걸 처음 봤다" — 근거를 다시 확인해달라는 요청.

### 재조사 결과
- 해당 문구는 라이선스 허여문이 아니라 **"이 소스 트리의 루트 LICENSE 파일"을 가리키는 상대 참조**였다.
- 원래 저장소인 **youngwanLEE/MPViT**의 루트 `LICENSE.md` = ETRI 듀얼 라이선스(비상업 GPLv3 / 상업은 ETRI 기술이전 계약).
- 그 파일이 **microsoft/unilm**으로 복사되면서 헤더 텍스트는 따라왔지만, 포인터가 가리키는 대상은 unilm 루트의 **MIT License (Copyright Microsoft Corporation)**로 바뀌었다.
- 즉 unilm 트리 안에서는 텍스트 그대로 읽어도 MIT를 가리킴 — 사용자의 기존 이해가 맞았다.
- 게다가 그 2개 파일(`ditod/backbone.py`, `ditod/__init__.py`)은 detectron2 전용이라 애초에 `dit_dino/`로 복제되지도 않는 파일들이었다.

계획서 §0을 "GPL 구역 분리" 설계에서 5줄짜리 사실관계 메모로 축소했고, 그에 따라 늘어났던 `LICENSE-THIRD-PARTY.md`, 구역별 `NOTICE`, grep 기반 분리 검증 등을 전부 되돌렸다.

### 추가 질문: "Apache-2.0 적용 범위는? GPL로부터 완전히 자유로울 수 있나?"
- Apache-2.0 §4(Redistribution)와 GPLv3 §2(*"make, run and propagate covered works that you do not convey, without conditions"*) 모두 **의무가 배포(conveying) 시점에만 발동**한다는 공통 구조를 확인.
- 개인 연구·비배포 전제에서는 두 라이선스 모두 실질적 의무가 없음.
- 결과 수치 공개, 논문 발표, 네트워크 서비스 제공은 "배포"에 해당하지 않음 (GPL·Apache-2.0 공통).

### 추가 질문: "git에 올릴 예정인데?"
- **private repo → 배포 아님, 여전히 의무 없음.**
- **public repo → conveying이므로 의무 발동.** 단, 필요한 것은 파일 몇 개뿐:
  - `dit_dino/LICENSE` (Apache-2.0, DINO 원본 그대로)
  - `baseline_dit_cascade/LICENSE` (MIT, microsoft/unilm 원문 — **주의**: WebFetch가 원문을 요약해버려 표준 MIT 정형 문구로 재구성함. 정확성이 중요하면 [microsoft/unilm/LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE)와 대조 필요)
  - 복제한 파일의 기존 저작권 헤더 보존, 수정한 파일에 변경 사실 한 줄 추가
  - NOTICE는 불필요 (DINO 원본에 NOTICE 파일 자체가 없음)
- `.gitignore`를 첫 커밋 전에 추가 (weight `.pth`, PubLayNet 이미지 등 대용량 파일 방지) — 라이선스와 무관하지만 더 실질적인 리스크로 계획에 포함.

## 4. 최종 산출물 — `test/` 디렉터리

```text
test/
├── README.md, .gitignore, requirements.txt
├── data/README.md           # PubLayNet 배치 규칙 (실데이터 커밋 안 함)
├── weights/README.md        # 사전학습 weight 다운로드 안내 (실파일 커밋 안 함)
├── common/                  # 프레임워크 비의존 — COCO json 채점/취합만
│   ├── coco_perclass_eval.py
│   ├── make_subset.py
│   └── collect_results.py
├── baseline_dit_cascade/    # DiT + Cascade R-CNN (detectron2)
│   ├── LICENSE (MIT)
│   ├── ditod/               # 무수정 복제
│   ├── publaynet_configs/   # 무수정 복제
│   ├── train_net.py         # 경로 인자화만 수정
│   ├── inference.py         # 무수정 복제
│   └── bench_speed_d2.py    # 신규
└── dit_dino/                # DiT + DINO (순수 PyTorch)
    ├── LICENSE (Apache-2.0)
    ├── dit/                 # 신규 패키지 — beit.py 무수정 복제 + dit_backbone.py + load_dit_weights.py
    ├── models/, datasets/, util/, config/, tools/, scripts/
    ├── main.py, engine.py   # 결과 dump, VRAM 로깅 추가
    └── bench_speed_dino.py  # 신규
```

### `dit/dit_backbone.py` — 핵심 구현
`beit.py`의 `patch_size==8` 분기가 만드는 연산 조합(2x ConvTranspose / Identity / MaxPool2 / MaxPool4)을, patch16 그대로의 backbone(사전학습 conv 가중치 유지)에 붙여 stride 8/16/32/64를 만든다. 입력을 64의 배수로 패딩(mask는 `True`로 패딩)한 뒤 `DiTMultiScale.forward_features()`를 그대로 호출하고, 4개 tap(layer3/5/7/11)을 DINO의 `NestedTensor` 인터페이스로 감싼다.

수작업으로 conv/maxpool 산술을 직접 추적해 두 입력 크기(64의 배수, 비배수 — 800×1333 등) 모두에서 기대 shape과 일치함을 검증했다 (torch가 없는 환경이라 실제 실행 대신 손 계산으로 대조).

## 5. 구현 중 발견해 고친 버그 3개

실제로 실행했다면 바로 막혔을 문제들 — 계획 단계가 아니라 구현 중 정적 검증/기능 테스트로 잡아냄.

1. **cp949 인코딩 크래시** — Windows 콘솔(cp949)에서 em-dash(—)가 포함된 `print`/`raise`/argparse 문자열을 출력하면 `UnicodeEncodeError`로 죽는다는 걸 실제 실행으로 발견. 실행 경로(출력·예외·argparse description)에 있는 모든 em-dash를 ASCII 하이픈으로 교체. 주석/docstring 안의 것은 실행 시 출력되지 않으므로 그대로 둠.
2. **`models/__init__.py` 의존성 오염** — `dit_backbone.py`를 `models/dit/`에 두면, import 시 `models/__init__.py` → `models/dino/dino.py` → 컴파일된 MSDeformAttn CUDA 확장까지 전부 끌려온다. DiT 백본만 검증하면 되는 스모크 테스트(`check_dit_backbone.py`)가 무관한 CUDA 빌드 여부에 막힐 뻔했다. `dit/`를 `models/` 밖으로 꺼내 독립 패키지로 재배치해 해결.
3. **config 병합 충돌** — `DINO_4scale_dit.py`에 `dataset_file = 'publaynet'`을 넣었더니, `main.py`의 cfg 병합 로직이 "이미 argparse 인자로 정의된 키"라며 `ValueError`를 던지는 걸 코드 추적으로 발견. config에서 제거하고 `--dataset_file publaynet`을 CLI로만 넘기도록 수정, `bench_speed_dino.py`의 내부 인자 구성에도 반영.

## 6. 검증한 것 / 못한 것

**검증함:**
- 무수정 복제 대상 전체가 원본과 byte-identical (diff 확인)
- 수정한 파일들의 diff가 의도한 변경과 정확히 일치
- `common/` 스크립트(make_subset.py, collect_results.py)를 합성 데이터로 실제 실행
- `dit_backbone.py`의 stride 산술을 손으로 직접 추적해 검증
- 모든 `.py` 파일 syntax 컴파일 통과

**못함 (환경 제약):**
- 이 세션 환경에 torch/detectron2/pycocotools가 없어 실제 GPU 실행/학습은 수행하지 못함
- `tools/check_dit_backbone.py`가 실제 환경에서의 첫 검증 관문

## 7. 다음 단계 (사용자 몫)

1. `test/`를 private repo로 먼저 업로드
2. `baseline_dit_cascade/LICENSE`(MIT) 텍스트를 [microsoft/unilm/LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE) 원문과 대조
3. GPU 환경에서 `python tools/check_dit_backbone.py --weights ...`로 스모크 테스트
4. 문제없으면 public 전환

---

*계획 원본: `C:\Users\shkim\.claude\plans\dit-dino-publaynet-pure-hellman.md`*
*구현 위치: `D:\sample\test\`*
