# Drone App - Versão com Joystick

## Mudanças Realizadas

### ❌ **Removido:**
- Todos os controles VR (hand tracking, laser controls, super hands)
- Botões interativos do drone (takeoff, land, battery, flip)
- Componentes VR específicos (moveable-screen, interactive-button, vr-debug)
- Dependências VR (WebXR, Super Hands, Hand Tracking)
- Configurações AR/VR da cena A-Frame

### ✅ **Adicionado:**

#### **Componente Joystick Controller**
- **Detecção automática** de joysticks conectados
- **Display em tempo real** do status do joystick
- **Controle da câmera** via analógico direito
- **Controle da tela de vídeo** via analógico esquerdo
- **Monitoramento de botões** pressionados

#### **Interface de Status**
- Texto dinâmico mostrando:
  - Status de conexão do joystick
  - Valores dos analógicos (X, Y)
  - Botões pressionados
  - Nome/ID do joystick conectado

## Como Usar

### **Conectar Joystick**
1. Conecte um joystick USB ou Bluetooth ao computador
2. Abra a aplicação no navegador
3. O status será atualizado automaticamente quando detectado

### **Controles do Joystick**
- **Analógico Direito:** Rotaciona a câmera (olhar ao redor)
- **Analógico Esquerdo:** Move a tela de vídeo na cena
- **Qualquer Botão:** Aparece no display de status

### **Controles do Teclado** (ainda funcionam)
- **WASD:** Mover tela de vídeo
- **Q/E:** Subir/descer tela
- **Setas:** Rotacionar tela
- **+/-:** Zoom in/out

## Arquivos Modificados

### **HTML (`templates/index.html`)**
- Removido: Controladores VR, botões de drone, painéis de status
- Adicionado: Texto de status do joystick
- Simplificado: Configuração da cena A-Frame

### **JavaScript (`static/js/vr-components.js`)**
- Removido: Componentes VR complexos
- Adicionado: `joystick-controller` component
- Mantido: `keyboard-controls` e `simple-vr-interaction` (simplificado)

### **CSS (`static/css/style.css`)**
- Removido: Estilos VR específicos
- Adicionado: Estilos para display de status do joystick
- Simplificado: Animações e estilos gerais

## Funcionalidades do Joystick

### **Detecção de Conexão**
```javascript
// Eventos automáticos
window.addEventListener('gamepadconnected', ...);
window.addEventListener('gamepaddisconnected', ...);
```

### **Leitura de Estado**
- **Analógicos:** `gamepad.axes[0-3]` (esquerdo X/Y, direito X/Y)
- **Botões:** `gamepad.buttons[0-n]` (pressed/value)
- **Atualização:** A cada 100ms para performance

### **Controles Implementados**
1. **Câmera (analógico direito):**
   - Rotação suave da visão
   - Inversão natural do eixo Y

2. **Tela de vídeo (analógico esquerdo):**
   - Movimento horizontal/vertical
   - Velocidade ajustável

## Estrutura Simplificada

```
vr-app/
├── templates/
│   └── index.html          # HTML limpo, foco no joystick
├── static/
│   ├── css/
│   │   └── style.css       # Estilos simplificados
│   └── js/
│       └── vr-components.js # Componente joystick
└── README_JOYSTICK.md      # Esta documentação
```

## Compatibilidade

### **Joysticks Testados:**
- Xbox Controller (USB/Bluetooth)
- PlayStation Controller (USB/Bluetooth)
- Joysticks genéricos USB
- Joysticks Bluetooth

### **Navegadores:**
- Chrome/Chromium (recomendado)
- Firefox
- Edge
- Safari (limitado)

## Debug

### **Console do Navegador:**
```javascript
// Verificar joysticks conectados
navigator.getGamepads();

// Ver eventos de conexão
// Mensagens automáticas no console quando conectar/desconectar
```

### **Display Visual:**
- Status em tempo real na tela
- Valores dos analógicos
- Botões pressionados
- Nome do dispositivo

## Próximos Passos

1. **Calibração:** Adicionar zona morta configurável
2. **Mapeamento:** Permitir remapear botões e analógicos
3. **Múltiplos Joysticks:** Suporte a mais de um joystick
4. **Salvamento:** Persistir configurações do usuário
5. **Vibração:** Implementar feedback háptico (se suportado)
