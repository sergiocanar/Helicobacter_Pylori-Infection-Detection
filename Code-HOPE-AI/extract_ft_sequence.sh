CUDA_VISIBLE_DEVICES=7 python extract_features.py \
    --mode finetuned_kf \
    --frames_dir data/GrastroHUN_Hpylori/frames \
    --labels_dir data/GrastroHUN_Hpylori/labels \
    --output_dir data/GrastroHUN_Hpylori/full_seq_features