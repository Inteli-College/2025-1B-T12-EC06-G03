---
title: API e Integração com Drone
sidebar_position: 5
---

# API e Integração com Drone

Documentação da integração entre a aplicação VR e o sistema de controle de drone.

## Visão Geral da Integração

A aplicação VR comunica-se com o drone através de uma API REST, enviando comandos e recebendo dados de telemetria e stream de vídeo.

### Arquitetura de Comunicação

```
VR App (Frontend) → Flask Backend → Drone Controller → Drone Físico
                 ←                ←                 ←
```

## Endpoints da API

### URL Base

```
http://10.140.0.11:5000
```

### Comandos Disponíveis

#### 1. Takeoff (Decolagem)

**Endpoint**: `POST /takeoff`

**Descrição**: Inicia a decolagem do drone

**Request**:
```javascript
fetch('http://10.140.0.11:5000/takeoff', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  }
});
```

**Response**:
```json
{
  "status": "success",
  "message": "Drone takeoff initiated",
  "timestamp": "2025-06-13T10:30:00Z"
}
```

#### 2. Land (Pouso)

**Endpoint**: `POST /land`

**Descrição**: Inicia o pouso do drone

**Request**:
```javascript
fetch('http://10.140.0.11:5000/land', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  }
});
```

**Response**:
```json
{
  "status": "success", 
  "message": "Drone landing initiated",
  "timestamp": "2025-06-13T10:35:00Z"
}
```

#### 3. Battery Status

**Endpoint**: `GET /battery`

**Descrição**: Obtém o nível atual da bateria

**Request**:
```javascript
fetch('http://10.140.0.11:5000/battery', {
  method: 'GET'
});
```

**Response**:
```json
{
  "status": "success",
  "battery_level": 85,
  "voltage": 12.6,
  "estimated_flight_time": "15 minutes",
  "timestamp": "2025-06-13T10:30:00Z"
}
```

#### 4. Flip Maneuver

**Endpoint**: `POST /flip`

**Descrição**: Executa manobra de flip

**Request**:
```javascript
fetch('http://10.140.0.11:5000/flip', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    direction: 'forward' // 'forward', 'backward', 'left', 'right'
  })
});
```

**Response**:
```json
{
  "status": "success",
  "message": "Flip maneuver executed",
  "direction": "forward",
  "timestamp": "2025-06-13T10:32:00Z"
}
```

### Stream de Vídeo

**Endpoint**: `GET /video_stream`

**Descrição**: Stream em tempo real da câmera do drone

**Implementação**:
```javascript
// No HTML
<video id="videoStream" autoplay loop muted crossorigin="anonymous" playsinline>
  <source src="http://10.140.0.11:5000/video_stream" type="video/mp4">
</video>
```

## Implementação no Frontend VR

### Componente de Comunicação

```javascript
AFRAME.registerComponent('drone-api', {
  schema: {
    baseUrl: { type: 'string', default: 'http://10.140.0.11:5000' },
    timeout: { type: 'number', default: 5000 }
  },
  
  init: function () {
    this.setupAPI();
  },
  
  setupAPI: function () {
    this.api = {
      takeoff: () => this.sendCommand('takeoff'),
      land: () => this.sendCommand('land'),
      flip: (direction = 'forward') => this.sendCommand('flip', { direction }),
      getBattery: () => this.sendCommand('battery', null, 'GET')
    };
  },
  
  sendCommand: async function (endpoint, data = null, method = 'POST') {
    const url = `${this.data.baseUrl}/${endpoint}`;
    
    try {
      const options = {
        method: method,
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: this.data.timeout
      };
      
      if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
      }
      
      const response = await fetch(url, options);
      const result = await response.json();
      
      this.handleResponse(endpoint, result);
      return result;
      
    } catch (error) {
      this.handleError(endpoint, error);
      throw error;
    }
  },
  
  handleResponse: function (endpoint, result) {
    console.log(`[DRONE-API] ${endpoint} response:`, result);
    
    // Emitir evento para outros componentes
    this.el.emit('drone-response', {
      endpoint: endpoint,
      result: result
    });
    
    // Feedback visual específico
    this.showNotification(`${endpoint.toUpperCase()}: ${result.message}`);
  },
  
  handleError: function (endpoint, error) {
    console.error(`[DRONE-API] ${endpoint} error:`, error);
    
    this.el.emit('drone-error', {
      endpoint: endpoint,
      error: error.message
    });
    
    this.showNotification(`Error: ${endpoint} failed`, 'error');
  },
  
  showNotification: function (message, type = 'info') {
    // Implementar notificação visual na cena VR
    const notification = document.createElement('a-text');
    notification.setAttribute('value', message);
    notification.setAttribute('color', type === 'error' ? 'red' : 'green');
    notification.setAttribute('position', '0 3 -2');
    
    this.el.sceneEl.appendChild(notification);
    
    // Remover após 3 segundos
    setTimeout(() => {
      this.el.sceneEl.removeChild(notification);
    }, 3000);
  }
});
```

### Integração com Botões Interativos

```javascript
// Atualização do componente interactive-button
AFRAME.registerComponent('interactive-button', {
  // ... schema existente ...
  
  executeAction: function () {
    const action = this.data.action;
    const droneAPI = document.querySelector('[drone-api]').components['drone-api'];
    
    switch (action) {
      case 'takeoff':
        droneAPI.api.takeoff();
        break;
      case 'land':
        droneAPI.api.land();
        break;
      case 'battery':
        droneAPI.api.getBattery().then(result => {
          this.displayBatteryInfo(result);
        });
        break;
      case 'flip':
        droneAPI.api.flip('forward');
        break;
      default:
        console.warn('Unknown action:', action);
    }
  },
  
  displayBatteryInfo: function (batteryData) {
    const message = `Battery: ${batteryData.battery_level}% (${batteryData.estimated_flight_time})`;
    this.showNotification(message);
  }
});
```

## Configuração do Stream de Vídeo

### Configuração no HTML

```html
<!-- Assets section -->
<a-assets>
  <video id="videoStream" 
         autoplay 
         loop 
         muted 
         crossorigin="anonymous" 
         playsinline
         width="1280" 
         height="720">
  </video>
</a-assets>

<!-- Video screen entity -->
<a-entity id="videoScreen" position="0 2 -2">
  <a-plane width="1.5" 
           height="1.125" 
           material="src: #videoStream; transparent: false; shader: flat">
  </a-plane>
</a-entity>
```

### Inicialização do Stream

```javascript
document.addEventListener('DOMContentLoaded', function () {
  setTimeout(() => {
    const video = document.getElementById('videoStream');
    if (video) {
      // Configurar source do stream
      video.src = 'http://10.140.0.11:5000/video_stream';
      
      // Event listeners para diagnóstico
      video.addEventListener('loadstart', () => {
        console.log('Video stream loading started');
      });
      
      video.addEventListener('canplay', () => {
        console.log('Video stream ready to play');
      });
      
      video.addEventListener('error', (e) => {
        console.error('Video stream error:', e);
        // Implementar fallback ou retry
      });
    }
  }, 1000);
});
```

## Tratamento de Erros

### Estados de Conexão

```javascript
AFRAME.registerComponent('connection-monitor', {
  init: function () {
    this.connectionState = 'disconnected'; // 'connected', 'connecting', 'disconnected', 'error'
    this.setupHealthCheck();
  },
  
  setupHealthCheck: function () {
    // Verificar conexão a cada 5 segundos
    this.healthCheckInterval = setInterval(() => {
      this.checkConnection();
    }, 5000);
  },
  
  checkConnection: async function () {
    try {
      const response = await fetch(`${this.data.baseUrl}/health`, {
        method: 'GET',
        timeout: 2000
      });
      
      if (response.ok) {
        this.setConnectionState('connected');
      } else {
        this.setConnectionState('error');
      }
    } catch (error) {
      this.setConnectionState('disconnected');
    }
  },
  
  setConnectionState: function (state) {
    if (this.connectionState !== state) {
      this.connectionState = state;
      this.updateUI(state);
      
      this.el.emit('connection-state-changed', {
        state: state
      });
    }
  },
  
  updateUI: function (state) {
    const statusColors = {
      connected: 'green',
      connecting: 'yellow', 
      disconnected: 'red',
      error: 'orange'
    };
    
    // Atualizar indicador visual na cena
    const statusIndicator = document.getElementById('connectionStatus');
    if (statusIndicator) {
      statusIndicator.setAttribute('material', 'color', statusColors[state]);
    }
  }
});
```

### Retry Logic

```javascript
AFRAME.registerComponent('api-retry', {
  schema: {
    maxRetries: { type: 'number', default: 3 },
    retryDelay: { type: 'number', default: 1000 }
  },
  
  sendWithRetry: async function (endpoint, data, method, attempt = 1) {
    try {
      return await this.sendCommand(endpoint, data, method);
    } catch (error) {
      if (attempt < this.data.maxRetries) {
        console.log(`Retry ${attempt + 1}/${this.data.maxRetries} for ${endpoint}`);
        
        // Delay progressivo
        const delay = this.data.retryDelay * attempt;
        await new Promise(resolve => setTimeout(resolve, delay));
        
        return this.sendWithRetry(endpoint, data, method, attempt + 1);
      } else {
        throw error;
      }
    }
  }
});
```

## Segurança e CORS

### Configuração CORS

Para permitir requisições do frontend VR para a API do drone:

```python
# No backend Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['http://localhost:5000', 'https://*.ngrok.io'])
```

### Headers de Segurança

```javascript
// Headers adicionais para segurança
const secureHeaders = {
  'Content-Type': 'application/json',
  'X-Requested-With': 'XMLHttpRequest',
  'Cache-Control': 'no-cache'
};
```

## Monitoramento e Logs

### Sistema de Logs

```javascript
AFRAME.registerComponent('api-logger', {
  init: function () {
    this.logs = [];
    this.setupLogging();
  },
  
  logAPICall: function (endpoint, method, data, response, duration) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      endpoint: endpoint,
      method: method,
      data: data,
      response: response,
      duration: duration,
      success: !response.error
    };
    
    this.logs.push(logEntry);
    
    // Manter apenas os últimos 100 logs
    if (this.logs.length > 100) {
      this.logs.shift();
    }
    
    // Log no console para debug
    console.log('[API-LOG]', logEntry);
  },
  
  getApiStats: function () {
    const stats = {
      totalCalls: this.logs.length,
      successRate: this.logs.filter(log => log.success).length / this.logs.length,
      averageResponseTime: this.logs.reduce((acc, log) => acc + log.duration, 0) / this.logs.length,
      errorCount: this.logs.filter(log => !log.success).length
    };
    
    return stats;
  }
});
```

## Conclusão

A integração com a API do drone é fundamental para o funcionamento da aplicação VR. O sistema implementa:

- **Comunicação robusta** com retry logic
- **Tratamento de erros** abrangente  
- **Monitoramento de conexão** em tempo real
- **Feedback visual** para o usuário
- **Logging** para diagnóstico

A arquitetura permite escalabilidade e manutenção fácil, com separação clara entre a lógica de comunicação e a interface VR.
