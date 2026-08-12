# 사전학습 Weight 배치

실제 `.pth` 파일은 이 저장소에 커밋하지 않는다 (`.gitignore` 참고, 약 340MB).

## 필요한 파일

```text
weights/dit-base-224-p16-500k-62d53a.pth
```

`baseline_dit_cascade/publaynet_configs/cascade/cascade_dit_base.yaml`의
`MODEL.WEIGHTS`와 `dit_dino/config/DINO/DINO_4scale_dit.py`의 `dit_pretrained_path`가
같은 파일을 가리켜야 한다 (계획서 6절 "동일 조건" 표) - 다운로드해서 이 폴더에 두고
필요하면 각 config의 경로를 맞춰 수정한다.

```bash
curl -L -o dit-base-224-p16-500k-62d53a.pth \
  https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth
```

## 출처 / 라이선스

[microsoft/unilm (DiT)](https://github.com/microsoft/unilm/tree/master/dit), **MIT**.
Weight 파일 자체의 라이선스는 이 저장소의 코드 라이선스와 별개다.
