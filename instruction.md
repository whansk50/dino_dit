## DiT+DINO 혼합 모델 구현
- 연구용으로 구현 목적
- 모델의 역할은 document layout detection
- git의 `unilm/dit`로부터 `object_detection`만을 남김. 필요한 경우 원본으로부터 코드를 가져오는 것을 권장.
- target dataset은 DiT에 근거해 publaynet을 사용할 예정
- pretrained 기반 finetune을 진행할 것이며, pretrained 모델을 가져와야 함

### 주요 과정 요구사항
- 별도의 폴더를 두고 그 아래에 구현
- `dit/`와 `DINO/`의 구성을 파악하고, DINO로 변경될 수 있는 부분에 대해 특정 flag를 이용해 신경망 switch가 가능하도록 수정해 적용
- I/O는 두 신경망 적용 시 기준 모두가 동일해야 함, 내부에서의 resize나 patch allocation 등은 논외
- 학습 시 사용할 hyperparameter는 yaml로 수정 가능 및 버전관리 포함 유지보수가 용이해야 함
- 시각화 툴은 MLflow를 적용

### 추가 요구사항
- 구현 후 markdown으로 workflow와 구성을 설명
- 현재 있는 `dit/`와 `DINO/`는 삭제 예정이므로, 해당 폴더를 참조하는 설계가 되어서는 안됨.
- publaynet은 huggingface를 사용해 다운로드할 예정이지만 이 다운로드를 코드에 직접 삽입하면 안됨
- 이외의 자율적인 판단에 의해 추가되는 기능은 핵심 기능이 아닌 경우(ex: 오류 방지 목적 등 차후에 적용해도 무리없는 경우)면 구현하지 말고 별도로 언급할 것.

### issue
- `dit/` 내부에 MPViT로부터 가져온 코드가 사용되고 있는 것으로 추정됨, MPViT는 GPL-3.0 license로, git public을 적용할 경우 license 관련 문제가 발생할 수 있어 관련 코드를 동일한 기능으로 재구축해야 함.
- `DINO/`는 Apache-2.0 license이므로 이에 주의해 코드를 수정할 필요