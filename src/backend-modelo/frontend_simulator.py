import socketio

sio = socketio.Client()

@sio.event
def connect():
    print("CLIENT: conectado ao servidor.")

@sio.event
def disconnect():
    print("CLIENT: desconectado do servidor.")

@sio.on("status", namespace="/ws/infer")
def status(data):
    print("CLIENT: STATUS:", data)

@sio.on("imagem_processada", namespace="/ws/infer")
def imagem_processada(data):
    print("CLIENT: IMAGEM:", data)

@sio.on("fim", namespace="/ws/infer")
def fim(data):
    print("CLIENT: FIM:", data)
    sio.disconnect()

@sio.on("error", namespace="/ws/infer")
def error(data):
    print("CLIENT: ERRO:", data)

@sio.on("results", namespace="/ws/infer")
def results(data):
    for result in data["results"]:
        print(f"ID: {result['id']}, Label: {result['label']}, Confidence: {result['confidence']}, Coords: {result['coords']}, Error: {result['error']}")

sio.connect(
    "http://localhost:5000",
    transports=["websocket"],
    namespaces=["/ws/infer"]
)

sio.emit(
    "infer_images",
    {"image_ids": [1, 2, 3]},
    namespace="/ws/infer"
)

sio.wait()
