# PubLayNet 데이터 배치

실제 이미지/annotation 파일은 이 저장소에 커밋하지 않는다 (`.gitignore` 참고).

## 배치 규칙

```text
data/publaynet/
├── train/          # PMCxxxx_00001.jpg ...
├── val/
├── train.json      # COCO detection 포맷
└── val.json
```

`baseline_dit_cascade/`와 `dit_dino/` 양쪽 모두 이 경로 하나(`../data/publaynet`)를
공유한다 (계획서 6절 "동일 조건" 표). 별도 사본을 두지 않는다 - 두 실험이 실제로
같은 GT를 보고 있는지 재확인하려면 `common/make_subset.py check`를 쓴다.

## Class 정의

| ID | Class |
|---:|---|
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

## 출처 / 라이선스

[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) (IBM), **CDLA-Permissive-1.0**.
데이터 자체의 라이선스는 이 저장소의 코드 라이선스(dit=MIT, dit_dino=Apache-2.0)와
별개다.
