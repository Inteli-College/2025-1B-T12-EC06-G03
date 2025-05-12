# load_data.py
from ultralytics.utils import YAML

# Ajuste o caminho conforme sua estrutura
yaml_path = "data/data.yaml"
data_config = YAML.load(yaml_path)

print("Configurações carregadas do dataset:")
for key, value in data_config.items():
    print(f"  {key}: {value}")
