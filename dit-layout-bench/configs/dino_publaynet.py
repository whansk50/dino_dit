# Internal bridge to IDEA DINO's SLConfig format.
# User-editable defaults belong in default.toml; the runtime forwards them here
# through --options so Cascade R-CNN and DINO share one configuration surface.
_base_ = ['../../DINO/config/DINO/DINO_4scale.py']

# PubLayNet uses category IDs 1..5. DINO expects max category ID + 1.
num_classes = 6
dn_labelbook_size = 6

backbone = 'dit_base_patch16'
dit_pretrained = None
dit_drop_path = 0.1
dit_use_checkpoint = True
return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 4

data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_norm_mean = [0.5, 0.5, 0.5]
data_norm_std = [0.5, 0.5, 0.5]
data_random_flip = 'horizontal'
strong_aug = False

num_queries = 300
num_select = 300
lr = 1e-4
lr_backbone = 1e-5
batch_size = 2
epochs = 12
lr_drop = 11
