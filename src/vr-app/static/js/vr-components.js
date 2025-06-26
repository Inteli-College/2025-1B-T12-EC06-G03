// VR Drone App JavaScript Components

// Keyboard Controls Component
AFRAME.registerComponent('keyboard-controls', {
  init: function () {
    this.bindEvents();
  },

  bindEvents: function () {
    this.keys = {};
    
    document.addEventListener('keydown', (evt) => {
      this.keys[evt.code] = true;
    });

    document.addEventListener('keyup', (evt) => {
      this.keys[evt.code] = false;
    });
  },

  tick: function () {
    let moved = false;
    const position = this.el.getAttribute('position');
    const rotation = this.el.getAttribute('rotation');
    const scale = this.el.getAttribute('scale') || { x: 1, y: 1, z: 1 };

    // Movement
    if (this.keys['KeyW']) { position.z -= 0.05; moved = true; }
    if (this.keys['KeyS']) { position.z += 0.05; moved = true; }
    if (this.keys['KeyA']) { position.x -= 0.05; moved = true; }
    if (this.keys['KeyD']) { position.x += 0.05; moved = true; }
    if (this.keys['KeyQ']) { position.y += 0.05; moved = true; }
    if (this.keys['KeyE']) { position.y -= 0.05; moved = true; }

    // Rotation
    if (this.keys['ArrowLeft']) { rotation.y += 2; moved = true; }
    if (this.keys['ArrowRight']) { rotation.y -= 2; moved = true; }
    if (this.keys['ArrowUp']) { rotation.x += 2; moved = true; }
    if (this.keys['ArrowDown']) { rotation.x -= 2; moved = true; }

    // Scale
    if (this.keys['Equal']) {
      scale.x = Math.min(scale.x + 0.01, 3);
      scale.y = Math.min(scale.y + 0.01, 3);
      scale.z = Math.min(scale.z + 0.01, 3);
      moved = true;
    }
    if (this.keys['Minus']) {
      scale.x = Math.max(scale.x - 0.01, 0.1);
      scale.y = Math.max(scale.y - 0.01, 0.1);
      scale.z = Math.max(scale.z - 0.01, 0.1);
      moved = true;
    }

    if (moved) {
      this.el.setAttribute('position', position);
      this.el.setAttribute('rotation', rotation);
      this.el.setAttribute('scale', scale);
    }
  }
});
AFRAME.registerComponent('gamepad-listener', {
  init: function () {
    this.statusText = null;
    this.lastUpdate = 0;
    this.updateInterval = 50; // Atualiza a cada 50ms
    this.speed = 0.05;

    this.el.addEventListener('loaded', () => {
      this.setupComponent();
    });

    setTimeout(() => {
      if (!this.statusText) {
        this.setupComponent();
      }
    }, 1000);

    console.log('Gamepad listener initializing...');
  },

  setupComponent: function () {
    this.statusText = document.querySelector('#statusText');

    if (this.statusText) {
      this.updateStatusText('Sistema de gamepad carregado - conecte um controle VR');
    }

    console.log('Gamepad listener ready');
  },

  updateStatusText: function (message) {
    if (this.statusText) {
      this.statusText.setAttribute('value', message);
    } else {
      this.statusText = document.querySelector('#statusText');
      if (this.statusText) {
        this.statusText.setAttribute('value', message);
      } else {
        console.log('Status text element not found:', message);
      }
    }
  },

  formatAnalogValue: function (value) {
    return Math.round(value * 100) / 100;
  },

  tick: function (time) {
    if (time - this.lastUpdate < this.updateInterval) {
      return;
    }
    this.lastUpdate = time;

    const gamepad = this.el.components["tracked-controls"]?.controller;

    if (!gamepad) {
      this.updateStatusText('Aguardando conexão do controle VR...');
      return;
    }

    try {
      let x = 0, y = 0;
      if (Array.isArray(gamepad.axes) && gamepad.axes.length >= 2) {
        [x, y] = gamepad.axes;
      }

      const leftX = this.formatAnalogValue(x || 0);
      const leftY = this.formatAnalogValue(y || 0);

      const pressedButtons = [];
      if (Array.isArray(gamepad.buttons)) {
        gamepad.buttons.forEach((button, index) => {
          if (button.pressed) {
            console.log(`Botão ${index} pressionado`);
            pressedButtons.push(index);
          }
        });
      }

      let statusMessage = 'Controle VR Ativo\n';
      statusMessage += `Joystick: (${leftX}, ${leftY})\n`;

      if (pressedButtons.length > 0) {
        statusMessage += `Botões: ${pressedButtons.join(', ')}\n`;
      }

      this.updateStatusText(statusMessage);

      const controlledElement =
        document.querySelector('#videoScreen') ||
        document.querySelector('#cube') ||
        document.querySelector('a-box');

      if (controlledElement && (Math.abs(x) > 0.1 || Math.abs(y) > 0.1)) {
        const position = controlledElement.getAttribute('position');
        if (position) {
          const newPosition = {
            x: position.x + (x * this.speed),
            y: position.y,
            z: position.z + (y * this.speed)
          };
          controlledElement.setAttribute('position', newPosition);
        }
      }

      if (Array.isArray(gamepad.axes) && gamepad.axes.length > 2) {
        const rightX = this.formatAnalogValue(gamepad.axes[2] || 0);
        const rightY = this.formatAnalogValue(gamepad.axes[3] || 0);

        const camera = document.querySelector('a-camera');
        if (camera && (Math.abs(rightX) > 0.1 || Math.abs(rightY) > 0.1)) {
          const currentRotation = camera.getAttribute('rotation');
          if (currentRotation) {
            const newRotation = {
              x: currentRotation.x + (rightY * -1),
              y: currentRotation.y + rightX,
              z: currentRotation.z
            };
            camera.setAttribute('rotation', newRotation);
          }
        }
      }

    } catch (error) {
      console.error('Erro ao ler dados do controle VR:', error);
      this.updateStatusText('Erro ao ler dados do controle VR:\n' + error.message);
    }
  }
});


// Simple VR Interaction Component (simplified for non-VR use)
AFRAME.registerComponent('simple-vr-interaction', {
  init: function () {
    console.log('Simple interaction component initialized on', this.el.id);
  }
});

// Joystick Debug Display Component
AFRAME.registerComponent('joystick-debug-display', {
  init: function () {
    this.gamepadIndex = -1;
    this.lastUpdate = 0;
    this.updateInterval = 50; // Update every 50ms for smooth display
    this.debugElement = null;
    this.isReady = false;
    
    // Wait for scene to be ready
    this.el.addEventListener('loaded', () => {
      this.setupComponent();
    });
    
    // Fallback - try to setup after a delay
    setTimeout(() => {
      if (!this.isReady) {
        this.setupComponent();
      }
    }, 1000);
    
    console.log('Joystick debug display initializing...');
  },

  setupComponent: function () {
    // Create debug display element if it doesn't exist
    this.debugElement = document.querySelector('#joystickDebug');
    if (!this.debugElement) {
      this.createDebugElement();
    }
    
    this.bindGamepadEvents();
    this.isReady = true;
    
    this.updateDebugDisplay('Aguardando conexão do joystick...');
    console.log('Joystick debug display ready');
  },

  createDebugElement: function () {
    // Create a text element for debug info
    const debugText = document.createElement('a-text');
    debugText.setAttribute('id', 'joystickDebug');
    debugText.setAttribute('position', '-2 3 -3');
    debugText.setAttribute('color', '#00ff00');
    debugText.setAttribute('font', 'monoid');
    debugText.setAttribute('width', '8');
    debugText.setAttribute('value', 'Carregando debug do joystick...');
    debugText.setAttribute('align', 'left');
    
    // Add to scene
    const scene = document.querySelector('a-scene');
    if (scene) {
      scene.appendChild(debugText);
      this.debugElement = debugText;
    }
  },

  bindGamepadEvents: function () {
    window.addEventListener('gamepadconnected', (e) => {
      console.log('Gamepad connected for debug:', e.gamepad);
      this.gamepadIndex = e.gamepad.index;
      this.updateDebugDisplay('Joystick conectado: ' + e.gamepad.id);
    });

    window.addEventListener('gamepaddisconnected', (e) => {
      console.log('Gamepad disconnected for debug:', e.gamepad);
      this.gamepadIndex = -1;
      this.updateDebugDisplay('Joystick desconectado - reconecte o controle');
    });
  },

  updateDebugDisplay: function (message) {
    if (this.debugElement) {
      this.debugElement.setAttribute('value', message);
    } else {
      // Try to find the element again
      this.debugElement = document.querySelector('#joystickDebug');
      if (this.debugElement) {
        this.debugElement.setAttribute('value', message);
      } else {
        console.log('Debug element not found:', message);
      }
    }
  },

  getGamepadState: function () {
    if (!navigator.getGamepads) {
      return null;
    }
    
    const gamepads = navigator.getGamepads();
    if (this.gamepadIndex >= 0 && gamepads[this.gamepadIndex]) {
      return gamepads[this.gamepadIndex];
    }
    return null;
  },

  formatValue: function (value) {
    return (Math.round(value * 1000) / 1000).toFixed(3);
  },

  tick: function (time) {
    if (!this.isReady) {
      return;
    }
    
    // Only update at specified intervals
    if (time - this.lastUpdate < this.updateInterval) {
      return;
    }
    this.lastUpdate = time;

    const gamepad = this.getGamepadState();
    if (!gamepad) {
      this.updateDebugDisplay('Nenhum joystick detectado\nConecte um controle para ver os valores');
      return;
    }

    try {
      // Get all axis values
      const axes = gamepad.axes;
      const buttons = gamepad.buttons;

      // Create detailed debug message
      let debugMessage = `=== JOYSTICK DEBUG ===\n`;
      debugMessage += `Nome: ${gamepad.id}\n`;
      debugMessage += `Index: ${gamepad.index}\n\n`;
      
      // Show axes values
      debugMessage += `EIXOS (${axes.length}):\n`;
      for (let i = 0; i < axes.length; i++) {
        const axisName = this.getAxisName(i);
        debugMessage += `${axisName}: ${this.formatValue(axes[i])}\n`;
      }
      
      // Show pressed buttons
      const pressedButtons = [];
      for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].pressed) {
          pressedButtons.push(`${i}(${this.formatValue(buttons[i].value)})`);
        }
      }
      
      debugMessage += `\nBOTÕES PRESSIONADOS:\n`;
      if (pressedButtons.length > 0) {
        debugMessage += pressedButtons.join(', ');
      } else {
        debugMessage += 'Nenhum';
      }
      
      debugMessage += `\n\nTOTAL BOTÕES: ${buttons.length}`;

      this.updateDebugDisplay(debugMessage);

    } catch (error) {
      console.error('Error in joystick debug tick:', error);
      this.updateDebugDisplay('Erro ao ler dados do joystick:\n' + error.message);
    }
  },

  getAxisName: function (index) {
    const axisNames = {
      0: 'Esq X  ',
      1: 'Esq Y  ',
      2: 'Dir X  ',
      3: 'Dir Y  ',
      4: 'L2/LT  ',
      5: 'R2/RT  ',
      6: 'D-Pad X',
      7: 'D-Pad Y'
    };
    return axisNames[index] || `Eixo ${index}`;
  }
});
