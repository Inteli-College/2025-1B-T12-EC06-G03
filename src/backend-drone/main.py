from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from djitellopy import Tello
import cv2, threading, os, time
import logging

# Configurações de streaming
FRAME_WIDTH = 640  # Largura reduzida para melhor desempenho
FRAME_HEIGHT = 480  # Altura reduzida para melhor desempenho
FPS_TARGET = 30  # Taxa de quadros alvo
JPEG_QUALITY = 70  # Qualidade JPEG (0-100)

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tello_app')

# Classe para gerenciar o streaming de vídeo
class VideoStreamManager:
    def __init__(self):
        self.tello = None
        self.frame = None
        self.last_frame_time = 0
        self.is_streaming = False
        self.lock = threading.Lock()
        self.frame_count = 0
        self.start_time = time.time()
        
    def connect_tello(self):
        try:
            self.tello = Tello()
            self.tello.connect()
            logger.info("Tello drone conectado com sucesso")
            self.tello.streamon()
            logger.info("Stream de vídeo do Tello iniciado")
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar Tello: {e}")
            return False
            
    def start_capture_thread(self):
        if not self.is_streaming and self.tello:
            self.is_streaming = True
            self.start_time = time.time()
            thread = threading.Thread(target=self._capture_frames, daemon=True)
            thread.start()
            logger.info("Thread de captura de frames iniciada")
            
    def _capture_frames(self):
        frame_reader = self.tello.get_frame_read()
        while self.is_streaming:
            try:
                # Controle de taxa de quadros
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < 1.0/FPS_TARGET:
                    time.sleep(max(0, (1.0/FPS_TARGET) - elapsed))
                
                # Capturar e processar frame
                original_frame = frame_reader.frame
                if original_frame is None:
                    time.sleep(0.01)
                    continue
                    
                # Redimensionar e otimizar o frame para streaming
                processed_frame = self._process_frame(original_frame)
                
                # Armazenar o frame processado de forma thread-safe
                with self.lock:
                    self.frame = processed_frame
                    self.last_frame_time = time.time()
                    self.frame_count += 1
                    
                    # Calcular FPS real a cada 100 frames
                    if self.frame_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        fps = self.frame_count / elapsed
                        logger.info(f"FPS atual: {fps:.2f}")
                    
            except Exception as e:
                logger.error(f"Erro na captura de frame: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        # Redimensionar frame para melhorar performance
        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Converter de BGR para RGB para corrigir as cores
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb_frame

    
    def get_jpeg_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            # Encode com parâmetros otimizados de compressão
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', self.frame, encode_param)
            return buffer

# Instanciar gerenciador de streaming
stream_manager = VideoStreamManager()
if stream_manager.connect_tello():
    stream_manager.start_capture_thread()

# Helper decorator para comandos
def tello_command(func):
    def wrapper(data=None):
        if stream_manager.tello is None:
            emit('response', {'command': func.__name__, 'status': 'error', 'message': 'Tello não inicializado'})
            return
        try:
            result = func(data) if data is not None else func()
            emit('response', {'command': func.__name__, 'status': 'ok', 'response': result})
        except Exception as e:
            logger.error(f"Erro executando comando {func.__name__}: {e}")
            emit('response', {'command': func.__name__, 'status': 'error', 'message': str(e)})
    return wrapper

# Video generator para streaming MJPEG
def generate_mjpeg():
    while True:
        try:
            jpeg_buffer = stream_manager.get_jpeg_frame()
            if jpeg_buffer is None:
                time.sleep(0.01)
                continue
                
            jpg_bytes = jpeg_buffer.tobytes()
            
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Erro no streaming MJPEG: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    logger.info(f"Página inicial solicitada por: {request.remote_addr}")
    return app.send_static_file('index.html')

@app.route('/video')
def video_feed():
    """
    Stream MJPEG via HTTP para navegadores ou clientes HTTP.
    """
    logger.info(f"Stream de vídeo solicitado por: {request.remote_addr}")
    return Response(generate_mjpeg(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO events para controle do drone
@socketio.on('connect')
def on_connect():
    logger.info(f"Cliente conectado via Socket.IO: {request.sid}")
    emit('response', {'message': 'Connected to Tello server'})

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Cliente desconectado do Socket.IO: {request.sid}")

@socketio.on('takeoff')
@tello_command
def takeoff(data=None):
    logger.info("Comando recebido: takeoff")
    return stream_manager.tello.takeoff()

@socketio.on('land')
@tello_command
def land(data=None):
    logger.info("Comando recebido: land")
    return stream_manager.tello.land()

@socketio.on('flip')
@tello_command
def flip(data):
    direction = data.get('direction')
    logger.info(f"Comando recebido: flip {direction}")
    valid_directions = ['l', 'r', 'f', 'b']
    if direction not in valid_directions:
        raise ValueError(f"Direção de flip inválida: {direction}")
    return stream_manager.tello.send_command_with_return(f'flip {direction}')


@socketio.on('move')
@tello_command
def move(data):
    direction = data.get('direction')
    distance = data.get('distance')
    logger.info(f"Comando recebido: move {direction} {distance}cm")

    # Mapeamento válido de direção
    valid_directions = ['forward', 'back', 'left', 'right', 'up', 'down']
    if direction not in valid_directions:
        raise ValueError(f"Direção inválida: {direction}")

    # Comando correto para o Tello
    return stream_manager.tello.send_command_with_return(f'{direction} {distance}')


@socketio.on('rotate')
@tello_command
def rotate(data):
    direction = data.get('direction')
    degree = data.get('degree')
    logger.info(f"Comando recebido: rotate {direction} {degree}°")
    if direction not in ['cw', 'ccw']:
        raise ValueError("Direção de rotação deve ser 'cw' ou 'ccw'")
    return stream_manager.tello.send_command_with_return(f'{direction} {degree}')


@socketio.on('battery')
@tello_command
def battery(data=None):
    logger.info("Comando recebido: battery")
    return stream_manager.tello.get_battery()

@socketio.on('rc_control')
def rc_control(data):
    if stream_manager.tello is None:
        emit('response', {'command': 'rc_control', 'status': 'error', 'message': 'Tello não conectado'})
        return
    
    try:
        lr = int(data.get('lr', 0))   # left/right
        fb = int(data.get('fb', 0))   # forward/backward
        ud = int(data.get('ud', 0))   # up/down
        yw = int(data.get('yw', 0))   # yaw (giro)
        
        logger.debug(f"rc_control recebido: lr={lr}, fb={fb}, ud={ud}, yw={yw}")
        stream_manager.tello.send_rc_control(lr, fb, ud, yw)

    except Exception as e:
        logger.error(f"Erro em rc_control: {e}")
        emit('response', {'command': 'rc_control', 'status': 'error', 'message': str(e)})


@socketio.on('palm_land')
@tello_command
def palm_land(data=None):
    logger.info("Comando recebido: palm_land (pouso na mão)")
    return stream_manager.tello.send_command_with_return('palm_land')

if __name__ == '__main__':
    # Desativar reloader do Flask para evitar problemas de porta em uso
    logger.info(f"Iniciando servidor de controle Tello em http://0.0.0.0:5000")
    logger.info(f"Configurações de stream: {FRAME_WIDTH}x{FRAME_HEIGHT}, FPS alvo: {FPS_TARGET}, Qualidade JPEG: {JPEG_QUALITY}%")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
