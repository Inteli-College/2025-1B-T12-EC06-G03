---
title: Backend Drone
sidebar_position: 3
---

# Backend Drone

O backend do drone é responsável por estabelecer a comunicação entre a interface do usuário e o drone Tello da DJI, facilitando o controle das operações do drone e o streaming de vídeo em tempo real.

## Tecnologias Utilizadas

- **Flask**: Framework web para Python
- **Flask-SocketIO**: Implementação de WebSockets para Flask
- **Flask-CORS**: Extensão para lidar com Cross-Origin Resource Sharing
- **djitellopy**: Biblioteca Python para comunicação com drones Tello
- **OpenCV (cv2)**: Biblioteca para processamento de imagens
- **Threading**: Módulo para gerenciamento de threads paralelas

## Componentes Principais

### 1. Servidor Web Flask

O aplicativo Flask serve como ponto central de comunicação, fornecendo:

```python
app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
```

- **Rota `/`**: Página inicial da aplicação
- **Rota `/video`**: Endpoint para streaming de vídeo MJPEG

### 2. VideoStreamManager

Classe responsável pelo gerenciamento do streaming de vídeo do drone:

```python
class VideoStreamManager:
    def __init__(self):
        self.tello = None
        self.frame = None
        self.is_streaming = False
        # ...
```

Principais funcionalidades:
- Conexão com o drone Tello
- Captura contínua de frames em thread separada
- Processamento e otimização de imagens para streaming
- Controle de FPS (frames por segundo)
- Conversão dos frames para formato JPEG

### 3. Controlador de Eventos WebSocket

Interface de WebSocket para comunicação bidirecional em tempo real:

```python
@socketio.on('connect')
def on_connect():
    # ...

@socketio.on('takeoff')
@tello_command
def takeoff(data=None):
    # ...
```

## Comandos do Drone

O backend implementa os seguintes comandos para controle do drone:

| Comando | Descrição | Parâmetros |
|---------|-----------|------------|
| `takeoff` | Decolar o drone | Nenhum |
| `land` | Pousar o drone | Nenhum |
| `flip` | Realizar um flip | `direction`: 'l' (esquerda), 'r' (direita), 'f' (frente), 'b' (traseira) |
| `move` | Mover em uma direção | `direction`: 'forward', 'back', 'left', 'right', 'up', 'down'<br>`distance`: distância em cm |
| `rotate` | Girar o drone | `direction`: 'cw' (horário), 'ccw' (anti-horário)<br>`degree`: ângulo em graus |
| `battery` | Verificar nível da bateria | Nenhum |
| `rc_control` | Controle remoto contínuo | `lr`: esquerda/direita (-100 a 100)<br>`fb`: frente/trás (-100 a 100)<br>`ud`: cima/baixo (-100 a 100)<br>`yw`: giro (-100 a 100) |
| `palm_land` | Pousar na palma da mão | Nenhum |

## Streaming de Vídeo

O streaming de vídeo é implementado através de:

1. Captura contínua dos frames da câmera do drone
2. Processamento e otimização para transmissão web
3. Conversão para formato MJPEG
4. Streaming via protocolo HTTP

Configurações padrão:
- Resolução: 640x480 pixels
- FPS alvo: 30 frames por segundo
- Qualidade JPEG: 70%

## Sistema de Log

O backend possui um sistema de log que registra todas as operações e possíveis erros:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tello_app')
```

## Inicialização do Servidor

O servidor é iniciado com as seguintes configurações:

```python
socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
```

- Host: `0.0.0.0` (acessível de qualquer IP na rede)
- Porta: `5000`
- Modo de depuração: Ativado
- Reloader automático: Desativado (para evitar problemas com as threads do streaming)

## Fluxo de Operação

1. O servidor inicializa e tenta conectar ao drone Tello
2. Após conexão bem-sucedida, inicia o streaming de vídeo
3. Quando um cliente se conecta, pode:
   - Visualizar o stream de vídeo via HTTP
   - Enviar comandos para o drone via WebSocket
   - Receber respostas e estados do drone

## Otimização e Performance

O código implementa várias otimizações para melhorar a performance:

- Redimensionamento dos frames para reduzir o consumo de banda
- Controle de FPS para estabilizar a taxa de quadros
- Compressão JPEG ajustável
- Sistema de threading para não bloquear o servidor principal
- Lock para acesso thread-safe aos frames

## Tratamento de Erros

Cada comando e operação possui tratamento de erros específico para:
- Falha na conexão com o drone
- Problemas durante a captura de vídeo
- Parâmetros inválidos em comandos
- Erros de comunicação

## Requisitos

Para executar este backend, é necessário:

1. Python 3.7+
2. Bibliotecas listadas em `requirements.txt`:
   - flask
   - flask-socketio
   - flask-cors
   - djitellopy
   - opencv-python
   - python-engineio
   - python-socketio

## Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar o servidor
python main.py
```

Após iniciar o servidor, o drone deve estar ligado e conectado à mesma rede Wi-Fi que o computador executando o backend.