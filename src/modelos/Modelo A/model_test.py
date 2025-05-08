from huggingface_hub import hf_hub_download

# faz download e cacheia em ~/.cache/huggingface/hub/…
local_path = hf_hub_download(
    repo_id    = "OpenSistemas/YOLOv8-crack-seg",      # repositório no HF
    filename   = "yolov8x/weights/best.pt",            # caminho dentro do repo
    # revision = "main",  # opcional: branch, tag ou commit
)
print("Peso disponível em:", local_path)


from ultralytics import YOLO

# 1) baixe o peso conforme acima e atribua a `local_path`
# 2) então carregue o modelo a partir desse caminho
model = YOLO(local_path)  

results = model.predict(
    source='data/processed/fissuras_termicas/FT58.PNG',  
    task="segment",  # Alternativamente, use "classify" para classificação
    device="cpu",    # GPU 0, ou use "cpu"
    save=True,
    conf = 0.099,  # confiança mínima
    show_conf=False,  # Não mostrar confiança
    show_labels=False,  # Não mostrar rótulos
    show_boxes=False  # Não mostrar bounding rectangles
)

print("Máscaras salvas em:", results[0].path)
