from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# Defina os diretórios de origem e destino
masks_dir_train = "data/crack_segmentation_dataset/train/masks"
out_dir_train   = "data/crack_segmentation_dataset/labels/train"

masks_dir_val   = "data/crack_segmentation_dataset/val/masks"
out_dir_val     = "data/crack_segmentation_dataset/labels/val"

# Número de classes (no seu caso, apenas fissura = 1)
classes = 1

# Converte máscaras de treino
convert_segment_masks_to_yolo_seg(masks_dir_train, out_dir_train, classes)
# Converte máscaras de validação
convert_segment_masks_to_yolo_seg(masks_dir_val,   out_dir_val,   classes)
