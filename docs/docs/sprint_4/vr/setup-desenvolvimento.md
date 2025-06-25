---
title: Setup de Desenvolvimento VR
sidebar_position: 1
---

# Setup de Desenvolvimento VR

&emsp;Este guia detalha como configurar o ambiente de desenvolvimento para a aplicação VR de controle de drone.

## Pré-requisitos

### Hardware

- **Computador**: Processador moderno com suporte a WebGL
- **Dispositivo VR** (opcional): Meta Quest 2/3, HTC Vive, Oculus Rift
- **Conexão de Rede**: Para comunicação com o drone

### Software

- **Python 3.8+**
- **Node.js 16+** (para ferramentas auxiliares)
- **Navegador Moderno**: Chrome, Edge ou Firefox com suporte WebXR

## Configuração do Ambiente

### 1. Clonar o Repositório

```bash
git clone <repository-url>
cd 2025-1B-T12-EC06-G03/src/vr-app
```

### 2. Configurar Ambiente Python

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Linux/Mac)
source .venv/bin/activate

# Ativar ambiente (Windows)
.venv\Scripts\activate

# Instalar dependências
pip install flask flask-cors
```

### 3. Configurar HTTPS (Necessário para WebXR)

&emsp;Para que as funcionalidades VR funcionem, é necessário servir a aplicação via HTTPS:

#### Opção 1: Certificado Local (Desenvolvimento)

```bash
# Instalar mkcert
brew install mkcert  # macOS
# ou
choco install mkcert  # Windows

# Criar certificados locais
mkcert -install
mkcert localhost 127.0.0.1 ::1
```

#### Opção 2: Túnel ngrok

```bash
# Instalar ngrok
npm install -g ngrok

# Executar aplicação
python app.py

# Em outro terminal, criar túnel HTTPS
ngrok http 5000
```

### 4. Configuração de Rede

&emsp;Editar o arquivo de configuração para ajustar endpoints do drone:

```javascript
// Em templates/index.html, localizar e ajustar:
const DRONE_API_BASE = 'http://10.140.0.11:5000';  // Ajustar conforme necessário
```

## Execução da Aplicação

### Modo Desenvolvimento

```bash
cd src/vr-app
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
python app.py
```

### Acesso à Aplicação

- **Local**: `https://localhost:5000`
- **Ngrok**: `https://xxxxx.ngrok.io` (URL fornecida pelo ngrok)

## Configuração de Dispositivos VR

### Meta Quest 2/3

1. **Habilitar Modo Desenvolvedor**:
   - Instalar app Oculus no smartphone
   - Criar organização de desenvolvedor
   - Habilitar modo desenvolvedor no headset

2. **Configurar Browser**:
   - Usar o browser nativo do Quest
   - Ou instalar Firefox Reality

3. **Conectar à Aplicação**:
   - Acessar URL HTTPS da aplicação
   - Permitir acesso a sensores VR

### Desktop (Desenvolvimento sem VR)

&emsp;Para desenvolvimento sem headset VR:

1. **Chrome**: Habilitar WebXR Emulator
   - Instalar extensão "WebXR API Emulator"
   - Configurar dispositivo virtual

2. **Firefox**: Habilitar flags WebXR
   - Acessar `about:config`
   - Habilitar `dom.vr.enabled`

## Ferramentas de Debug

### Console de Debug VR

&emsp;A aplicação inclui um sistema de debug integrado:

```javascript
// Logs automáticos são exibidos no console do browser
// Acesse via F12 > Console
```

### Debug Remoto (Quest)

```bash
# Conectar Quest via USB
adb devices

# Abrir debug remoto
chrome://inspect/#devices
```

### A-Frame Inspector

&emsp;Pressione `Ctrl + Alt + I` na aplicação para abrir o inspetor 3D do A-Frame.

## Estrutura de Desenvolvimento

### Organização do Código

```
vr-app/
├── app.py                 # Servidor Flask
├── templates/
│   ├── index.html        # Interface principal
│   └── index_simple.html # Interface simplificada
├── static/
│   ├── css/              # Estilos customizados
│   ├── js/               # Scripts auxiliares
│   └── assets/           # Recursos 3D
└── requirements.txt      # Dependências Python
```

### Convenções de Código

#### JavaScript/A-Frame

```javascript
// Componentes customizados
AFRAME.registerComponent('nome-componente', {
  schema: {
    // Propriedades configuráveis
  },
  
  init: function() {
    // Inicialização
  },
  
  tick: function() {
    // Loop de atualização
  }
});
```

#### HTML/A-Frame

```html
<!-- Entidades interativas devem ter classe .interactive -->
<a-entity class="interactive grabbable"
          custom-component="property: value">
</a-entity>
```

## Troubleshooting Comum

### WebXR não detectado

**Problema**: "WebXR not supported"

**Soluções**:
- Verificar se está usando HTTPS
- Atualizar navegador
- Verificar suporte do dispositivo

### Controladores não aparecem

**Problema**: Controladores VR não são detectados

**Soluções**:
- Recarregar página
- Verificar permissões do browser
- Testar com hand tracking

### Performance baixa

**Problema**: Aplicação lenta em VR

**Soluções**:
- Reduzir qualidade gráfica
- Otimizar número de entidades
- Verificar recursos do sistema

### Conexão com Drone falha

**Problema**: Não consegue enviar comandos

**Soluções**:
- Verificar conectividade de rede
- Confirmar endpoint do drone
- Testar API manualmente

## Próximos Passos

Após configurar o ambiente:

1. Testar funcionalidades básicas
2. Configurar conexão com drone real
3. Implementar melhorias de UX
4. Otimizar performance para dispositivo alvo

## Recursos Adicionais

- [A-Frame Documentation](https://aframe.io/docs/)
- [WebXR Specs](https://www.w3.org/TR/webxr/)
- [Meta Quest Developer Guide](https://developer.oculus.com/documentation/web/browser-intro/)
