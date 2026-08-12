# DiT + DINO PubLayNet 벤치마크용 config.
# DINO_4scale.py(-> coco_transformer.py)를 그대로 상속하고, 아래 값만 덮어쓴다.
_base_ = ['DINO_4scale.py']

# ---- 백본: ResNet 대신 DiT-base ----
# DiT는 Joiner(backbone, position_embedding)로 감싸져 파라미터 이름에 "backbone."이
# 그대로 붙으므로 param_dict_type='default'(get_param_dicts.py)의 substring 매칭이
# 별도 설정 없이 그대로 통한다.
backbone = 'dit_base_patch16'
dit_pretrained_path = '../weights/dit-base-224-p16-500k-62d53a.pth'
dit_img_size = [224, 224]
dit_pos_type = 'abs'
dit_drop_path = 0.1
dit_use_checkpoint = True
dit_out_layers = ['layer3', 'layer5', 'layer7', 'layer11']
dit_pyramid_strides = [8, 16, 32, 64]  # B2 ablation 시 [16, 32, 64]로 교체
return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 4

# ---- 데이터 / 클래스: PubLayNet ----
# dataset_file/coco_path는 여기서 설정하지 않는다 - main.py의 get_args_parser()가 이미
# 정의한 argparse 인자라서 cfg 파일에 같은 키를 두면 main.py의 병합 로직이
# "Key dataset_file can used by args only" ValueError를 던진다. 반드시
# --dataset_file publaynet --coco_path <root> 로 CLI에서 넘긴다 (scripts/*.sh 참고).
num_classes = 6  # PubLayNet category_id 1~5 + 미사용 0 슬롯 (계획서 조사 5)
dn_labelbook_size = 6
data_norm_mean = [0.5, 0.5, 0.5]  # DiT 사전학습 정규화에 맞춘다 (계획서 조사 4)
data_norm_std = [0.5, 0.5, 0.5]

# ---- detector: query 수 축소 (계획서 §6) ----
num_queries = 300
num_select = 300

# ---- 최적화 (계획서 §9, §5) ----
lr = 1e-4
lr_backbone = 1e-5
batch_size = 2
epochs = 3
lr_drop = 2
