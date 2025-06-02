---
title: Backend Modelo
sidebar_position: 1
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Backend do modelo
&emsp;Na sprint 2, foi desenvolvida uma aplicação simples que utilizava a ferramenta *Streamlit* para rodar os modelos de classificação de fissuras em uma interface gráfica amigável. Entretanto, essa aplicação foi feita apenas como uma maneira de prototipar algo rápido e mostrar resultados mais concretos aos parceiros de projeto. Dessa forma, para a aplicação final, os modelos de classificação estarão embutidos, e funcionarão de modo que, dentro do site, será possível enviar imagens e receber as classificações diretamente. Essa seção da documentação aborda o desenvolvimento do backend dessa parte da aplicação.

## Estrutura Geral

O backend é construído sobre o microframework Flask, utilizando extensões para lidar com banco de dados (Flask-SQLAlchemy) e comunicação em tempo real (Flask-SocketIO). A estrutura principal da aplicação é definida no arquivo `app/__init__.py`:

```python
# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

db = SQLAlchemy()
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config') 
    app.config['SECRET_KEY'] = 'secret!' 

    db.init_app(app) 
    socketio.init_app(app) 

    from app.routes.images import images_bp
    from app.routes.health import health_bp
    from app.routes.inference import inference_bp
    from app.routes.websocket_routes import socketio_bp

    app.register_blueprint(images_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    app.register_blueprint(inference_bp, url_prefix='/api/v1')
    app.register_blueprint(socketio_bp) 

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()

    return app
```

Esta função `create_app` atua como uma factory, inicializando a aplicação Flask, configurando as extensões e registrando os diferentes módulos (Blueprints) que contêm as rotas da API e a lógica do WebSocket.



## Tecnologias Utilizadas

O backend da aplicação foi construído utilizando um conjunto de tecnologias modernas e eficientes para garantir robustez, escalabilidade e a capacidade de processamento de imagens em tempo real. A seguir, detalhamos as principais ferramentas e bibliotecas empregadas:

### Framework Web: Flask

O [Flask](https://flask.palletsprojects.com/) é um microframework web para Python, conhecido por sua simplicidade, flexibilidade e extensibilidade. Ele fornece a estrutura base para a criação das rotas da API, gerenciamento de requisições HTTP e a organização geral do projeto. A escolha do Flask permite um desenvolvimento ágil e a fácil integração com outras bibliotecas.

```python
# Exemplo de inicialização do Flask em app/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)
    return app
```

### Banco de Dados e ORM: Flask-SQLAlchemy

Para a persistência de dados, como informações sobre as imagens enviadas e seus status, foi utilizado o [SQLAlchemy](https://www.sqlalchemy.org/), um poderoso ORM (Object-Relational Mapper) para Python, integrado ao Flask através da extensão [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/). Isso abstrai as interações com o banco de dados SQL, permitindo definir modelos de dados como classes Python e realizar operações de forma mais intuitiva e segura.

### Comunicação em Tempo Real: Flask-SocketIO

Para a funcionalidade de inferência de imagens, onde o cliente envia IDs de imagens e aguarda os resultados da classificação, foi implementada a comunicação via WebSockets utilizando a extensão [Flask-SocketIO](https://flask-socketio.readthedocs.io/). Isso permite uma comunicação bidirecional e em tempo real entre o cliente (frontend) e o servidor (backend), ideal para notificar o progresso e enviar os resultados da classificação de forma assíncrona, sem a necessidade de polling constante.

```python
# Exemplo de inicialização e uso em app/__init__.py e routes/websocket_routes.py
from flask_socketio import SocketIO, emit, Namespace

socketio = SocketIO(cors_allowed_origins="*")

# Em app/__init__.py
def create_app():
    socketio.init_app(app)

# Em app/routes/websocket_routes.py
class InferenceNamespace(Namespace):
    def on_connect(self):
        print("Cliente conectado.")

    def on_infer_images(self, data):
        emit("results", {"results": results})

socketio.on_namespace(InferenceNamespace("/ws/infer"))
```

## Rotas e Funcionalidades

O backend expõe funcionalidades através de uma API RESTful e de um endpoint WebSocket para comunicação em tempo real. A seguir, detalhamos as principais rotas e eventos disponíveis.

### API RESTful (prefixo `/api/v1`)

As rotas RESTful são responsáveis pelo gerenciamento de imagens e pela verificação do estado da aplicação. Elas são organizadas em Blueprints do Flask.

#### Inferência (Alternativa REST - `inference_bp`)

*   **Endpoint:** (Ex: `POST /api/v1/inference/{image_id}`)
    *   **Descrição:** Poderia existir uma rota REST para solicitar a inferência de uma única imagem de forma síncrona. No entanto, a implementação principal e recomendada para inferência em lote utiliza WebSockets (ver seção seguinte) devido à natureza potencialmente demorada do processo.

### Comunicação via WebSocket (`socketio_bp`)

Para a classificação das imagens, que pode ser um processo mais longo, utiliza-se WebSockets para comunicação assíncrona e em tempo real entre o cliente e o servidor. Isso evita timeouts de requisições HTTP e permite que o servidor envie atualizações de status e resultados conforme são processados.

*   **Namespace:** `/ws/infer`
*   **Descrição:** Este namespace é dedicado ao processo de inferência de imagens.

#### Eventos

1.  **`connect` (Cliente -> Servidor -> Cliente)**
    *   **Descrição:** O cliente estabelece uma conexão WebSocket com o namespace `/ws/infer`. O servidor registra a conexão.
    *   **Log no Servidor:** `Cliente conectado.`

2.  **`infer_images` (Cliente -> Servidor)**
    *   **Descrição:** O cliente envia uma lista de IDs de imagens que deseja classificar.
    *   **Payload (Exemplo):**
        ```json
        {
          "image_ids": [1, 5, 12]
        }
        ```
    *   **Processo no Servidor:**
        *   Valida se `image_ids` é uma lista de inteiros.
        *   Se inválido, emite um evento `error`.
        *   Busca as imagens correspondentes no banco de dados.
        *   Emite um evento `status` inicial.
        *   Para cada imagem:
            *   Baixa a imagem da URL configurada (`IMG_URL_PREFIX` + `caminho_arquivo`).
            *   Executa o pré-processamento (OpenCV).
            *   Realiza a detecção da fissura (YOLO).
            *   Se detectada, recorta a região e classifica (CNN PyTorch).
            *   Atualiza o status `processada` da imagem no banco de dados.
            *   Coleta o resultado (label, confiança, coordenadas) ou erro.
            *   Remove a imagem baixada localmente.
        *   Após processar todas as imagens, emite um evento `results` com todos os resultados.
        *   Emite um evento `fim`.
        *   Desconecta o cliente.

3.  **`status` (Servidor -> Cliente)**
    *   **Descrição:** Enviado pelo servidor para informar o cliente sobre o progresso do processamento.
    *   **Payload (Exemplo):**
        ```json
        {
          "message": "3 imagens recebidas. Processando..."
        }
        ```

4.  **`results` (Servidor -> Cliente)**
    *   **Descrição:** Enviado pelo servidor contendo os resultados da classificação para o lote de imagens solicitado.
    *   **Payload (Exemplo):**
        ```json
        {
          "results": [
            {
              "id": 1,
              "caminho": "http://storage.example.com/uploads/img1.jpg",
              "label": "Transversal",
              "confidence": 0.95,
              "coords": {"x1": 100, "y1": 150, "x2": 300, "y2": 200},
              "error": null
            },
            {
              "id": 5,
              "caminho": "http://storage.example.com/uploads/img5.png",
              "label": null,
              "confidence": null,
              "coords": null,
              "error": "Erro ao baixar a imagem."
            },
            {
              "id": 12,
              "caminho": "http://storage.example.com/uploads/img12.jpeg",
              "label": "Nenhuma fissura detectada",
              "confidence": 0.0,
              "coords": null,
              "error": null
            }
          ]
        }
        ```

5.  **`error` (Servidor -> Cliente)**
    *   **Descrição:** Enviado pelo servidor em caso de erro na validação da requisição do cliente.
    *   **Payload (Exemplo):**
        ```json
        {
          "error": "'image_ids' deve ser uma lista de inteiros"
        }
        ```

6.  **`fim` (Servidor -> Cliente)**
    *   **Descrição:** Indica que o processamento do lote de imagens foi concluído.
    *   **Payload (Exemplo):**
        ```json
        {
          "message": "Processamento completo."
        }
        ```

7.  **`disconnect` (Cliente -> Servidor -> Cliente)**
    *   **Descrição:** Ocorre quando a conexão é encerrada, seja pelo cliente ou pelo servidor (após o evento `fim`).
    *   **Log no Servidor:** `Cliente desconectado.`

### Fluxo de Inferência (WebSocket)

1.  O Frontend estabelece uma conexão WebSocket com `/ws/infer`.
2.  O Frontend envia o evento `infer_images` com a lista de IDs das imagens a serem classificadas.
3.  O Backend recebe a lista, valida, e envia um evento `status` confirmando o início.
4.  O Backend processa cada imagem sequencialmente (download, detecção, classificação).
5.  Após processar todas as imagens, o Backend envia o evento `results` contendo os resultados para cada imagem (incluindo possíveis erros).
6.  O Backend envia o evento `fim` para sinalizar o término.
7.  O Backend encerra a conexão WebSocket.

### Conclusão

&emsp;Este documento detalhou a arquitetura e as funcionalidades do backend desenvolvido para a classificação de fissuras em imagens. Partindo de um protótipo inicial em Streamlit, a solução evoluiu para um sistema robusto baseado em Flask, projetado para integração direta com a aplicação web final.
A estrutura modularizada com Blueprints facilita a manutenção e a expansão futura do backend. A comunicação via WebSocket foi escolhida estrategicamente para lidar com o processamento potencialmente demorado das imagens, proporcionando uma melhor experiência ao usuário ao fornecer feedback em tempo real sobre o status da classificação.