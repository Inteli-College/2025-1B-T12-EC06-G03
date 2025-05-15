---
title: Sprint 2
sidebar_position: 0
---

# Sprint 2

## Introdução

A Sprint 2 do projeto Athenas foca na implementação de funcionalidades essenciais para o controle remoto de drones, incluindo a configuração do aplicativo, a arquitetura técnica e o sistema de comunicação. Além disso, também criamos o modelo e o front-end do sistema. O objetivo é garantir uma experiência de usuário fluida e responsiva, permitindo o controle eficiente dos drones em tempo real.
A seguir, apresentamos uma visão geral das principais funcionalidades e componentes desenvolvidos nesta sprint.

## Funcionalidades Implementadas

### Modelo no Streamlit
- **Classificação de Rachaduras e Fissuras**: Implementação de um modelo de aprendizado de máquina no Streamlit para classificar rachaduras e fissuras em imagens. O modelo utiliza técnicas de visão computacional para analisar as imagens enviadas pelo usuário e fornecer uma classificação precisa.
- **Interface Interativa**: Criação de uma interface amigável no Streamlit para upload de imagens e exibição dos resultados da classificação.
- **Treinamento e Validação**: O modelo foi treinado com um conjunto de dados de imagens de rachaduras e fissuras, garantindo alta precisão e confiabilidade nos resultados.

#### Como Rodar o Modelo no Streamlit
1. **Clone o repositório**: 
```bash
git clone <URL do repositório>
```
2. **Navegue até o diretório do projeto**: 
```bash
cd <diretório do projeto>/src/modelos/streamlit
```
3. **Crie um ambiente virtual**: 
```bash
python -m venv venv
```
4. **Ative o ambiente virtual**: 
   - No Windows: 
```bash
venv\Scripts\activate
```
   - No Linux/Mac: 
```bash
source venv/bin/activate
```
5. **Instale as dependências**: 
```bash 
pip install -r requirements.txt
```
6. **Inicie o Streamlit**: 
```bash
streamlit run app.py
```
7. **Acesse o aplicativo**: Abra o navegador e acesse `http://localhost:8501`
8. **Faça o upload de uma imagem**: Utilize a interface para fazer o upload de uma imagem e aguarde a classificação.


### Front-end
- **Componentes Reutilizáveis**: Desenvolvimento de componentes reutilizáveis para otimizar a manutenção e escalabilidade do código.
- **Design Responsivo**: Implementação de um layout responsivo para garantir uma experiência consistente em diferentes dispositivos.
- **Integração com o Drone**: Implementação de uma funcionalidade que permite visualizar a câmera do drone em tempo real diretamente no aplicativo, além de controlá-lo remotamente através da interface de controle.
- **Streaming de Vídeo**: Configuração de um sistema de streaming para exibir a visão do drone no aplicativo, garantindo baixa latência e alta qualidade de imagem.
- **Controle em Tempo Real**: Sincronização dos comandos enviados pelo aplicativo com as ações do drone, proporcionando uma experiência de controle precisa e responsiva.

#### Como Rodar o Front-end

1. **Clone o repositório**: 
```bash 
git clone <URL do repositório>
```
2. **Navegue até o diretório do projeto**: 
```bash
cd <diretório do projeto>/src/frontend
```
3. **Instale as dependências**: 
```bash
npm install
```
4. **Inicie o servidor**: 
```bash
npm start
```
5. **Acesse o aplicativo**: Abra o navegador e acesse `http://localhost:3000`

### Back-end do Drone
- **Servidor Flask**: Implementação de um servidor Flask para gerenciar a comunicação entre o aplicativo e o drone.
- **API RESTful**: Criação de endpoints para receber comandos do aplicativo e enviar dados de telemetria do drone.
- **Socket.IO**: Implementação de comunicação em tempo real entre o servidor e o aplicativo, permitindo o envio de comandos e recebimento de dados de telemetria instantaneamente.
- **Gerenciamento de Conexões**: Implementação de um sistema robusto para gerenciar múltiplas conexões simultâneas, garantindo que o servidor possa lidar com vários usuários ao mesmo tempo.

#### Como Rodar o Back-end do Drone
1. **Clone o repositório**: 
```bash
git clone <URL do repositório>
```
2. **Navegue até o diretório do projeto**: 
```bash
cd <diretório do projeto>/src/backend-drone
```
3. **Crie um ambiente virtual**: 
```bash
python -m venv venv
```
4. **Ative o ambiente virtual**: 
   - No Windows: 
```bash
venv\Scripts\activate
```
   - No Linux/Mac: 
```bash
source venv/bin/activate
```
5. **Instale as dependências**: 
```bash
pip install -r requirements.txt
```
6. **Inicie o servidor**: 
```bash
python main.py
```
7. **Acesse o servidor**: Abra o navegador e acesse `http://localhost:5000`

### App de Controle do Drone
- **Sistema de Controle**: Implementação dos controles de joystick e botões para interação com o drone.
- **Configuração do App**: Tela de configuração para definir o IP e a porta do servidor do drone.
- **Sistema de Comunicação**: Estabelecimento de comunicação entre o aplicativo e o servidor do drone utilizando Socket.IO e HTTP.

#### Como Rodar o App de Controle do Drone
1. **Clone o repositório**: 
```bash
git clone <URL do repositório>
```
2. **Navegue até o diretório do projeto**: 
```bash
cd <diretório do projeto>/src/app/athenas
```
3. **Instale as dependências**: 
```bash
flutter pub get
```
4. **Inicie o aplicativo**: 
```bash
flutter run
```
5. **Conecte-se ao servidor**: Insira o IP e a porta do servidor do drone na tela de configuração.

## Conclusão
A Sprint 2 do projeto Athenas trouxe avanços significativos na implementação de funcionalidades essenciais para o controle remoto de drones. Com a integração do modelo de aprendizado de máquina, o front-end responsivo e o back-end robusto, estamos um passo mais perto de alcançar nossos objetivos. As próximas etapas envolverão testes adicionais e melhorias com base no feedback dos usuários.