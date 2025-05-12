import os
import shutil
import random
from pathlib import Path

# Defina os caminhos base
base_dir = Path('data')
raw_image_dir = base_dir / 'raw'  # Diretório original das imagens
label_dir = base_dir / 'labels'  # Diretório original das labels

# Diretórios de saída do dataset
output_dir = base_dir / 'dataset'
train_image_dir = output_dir / 'images' / 'train'
val_image_dir = output_dir / 'images' / 'val'
train_label_dir = output_dir / 'labels' / 'train'
val_label_dir = output_dir / 'labels' / 'val'

# Crie os diretórios de saída
for directory in [train_image_dir, val_image_dir, train_label_dir, val_label_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Parâmetros de divisão
train_ratio = 0.8
seed = 42  # Para reproducibilidade
random.seed(seed)

# Lista todas as categorias
categories = ['fissuras_de_retracao', 'fissuras_termicas']

# Processamento principal
for category in categories:
    # Caminhos das imagens e labels originais
    image_category_dir = raw_image_dir / category
    label_category_dir = label_dir / category
    
    # Listar e validar arquivos
    images = list(image_category_dir.glob('*.PNG'))
    random.shuffle(images)
    
    # Verificar correspondência entre imagens e labels
    valid_images = []
    for img_path in images:
        label_path = label_category_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_images.append(img_path)
        else:
            print(f"Aviso: Label faltando para {img_path.name}")
    
    # Divisão treino/validação
    split_idx = int(len(valid_images) * train_ratio)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    # Função para copiar arquivos
    def copy_files(file_list, img_dest, lbl_dest):
        for img_path in file_list:
            label_path = label_category_dir / f"{img_path.stem}.txt"
            
            # Copiar imagem
            shutil.copy(img_path, img_dest / img_path.name)
            
            # Copiar label
            shutil.copy(label_path, lbl_dest / label_path.name)
    
    # Copiar arquivos
    copy_files(train_images, train_image_dir, train_label_dir)
    copy_files(val_images, val_image_dir, val_label_dir)

print("Divisão concluída com sucesso!")
print(f"Arquivos de treino: {len(train_images)}")
print(f"Arquivos de validação: {len(val_images)}")