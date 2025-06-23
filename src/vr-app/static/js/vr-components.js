// VR Drone App JavaScript Components

// Moveable Screen Component
AFRAME.registerComponent('moveable-screen', {
  onGripDown: function (evt) {
    console.log('Screen grabbed!');
  },

  onGripUp: function (evt) {
    console.log('Screen released!');
  },

  onMouseDown: function (evt) {
    this.el.setAttribute('material', 'color', '#88ff88');
  },

  onMouseUp: function (evt) {
    this.el.setAttribute('material', 'color', '#ffffff');
  },

  onTrackpadDown: function (evt) {
    this.currentScale = 1;
  },

  addGrabHandles: function () {
    this.el.appendChild(border);
  },

  tick: function () {
    // Movement logic can be added here
  }
});

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

// Interactive Button Component
AFRAME.registerComponent('interactive-button', {
  schema: {
    action: { type: 'string', default: 'default' },
    url: { type: 'string', default: 'http://10.140.0.11:5000' },
    label: { type: 'string', default: 'Button' }
  },

  init: function () {
    this.isGrabbed = false;
    this.originalPosition = this.el.getAttribute('position');
    this.originalRotation = this.el.getAttribute('rotation');
    this.originalScale = this.el.getAttribute('scale') || { x: 1, y: 1, z: 1 };
    this.isPressed = false;

    // VR Events - Super Hands
    this.el.addEventListener('grab-start', this.onGrabStart.bind(this));
    this.el.addEventListener('grab-end', this.onGrabEnd.bind(this));
    this.el.addEventListener('drag-move', this.onDragMove.bind(this));
    this.el.addEventListener('hover-start', this.onHoverStart.bind(this));
    this.el.addEventListener('hover-end', this.onHoverEnd.bind(this));

    // Direct controller events
    this.el.addEventListener('triggerdown', this.onTriggerDown.bind(this));
    this.el.addEventListener('triggerup', this.onTriggerUp.bind(this));

    // Interaction events
    this.el.addEventListener('click', this.onClick.bind(this));
    this.el.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.el.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.el.addEventListener('gripdown', this.onGripDown.bind(this));
    this.el.addEventListener('gripup', this.onGripUp.bind(this));

    // Visual feedback colors
    this.defaultColor = '#4CAF50';
    this.hoverColor = '#45a049';
    this.pressedColor = '#3d8b40';
    this.grabbedColor = '#2196F3';

    this.el.setAttribute('material', 'color', this.defaultColor);
  },

  onGrabStart: function (evt) {
    this.isGrabbed = true;
    this.grabbingController = evt.detail.hand;
    this.el.setAttribute('material', 'color', this.grabbedColor);
    console.log('Button grabbed with VR controller!');
  },

  onGrabEnd: function (evt) {
    this.isGrabbed = false;
    this.grabbingController = null;
    this.el.setAttribute('material', 'color', this.defaultColor);
    this.executeAction();
    this.animatePress();
  },

  onDragMove: function (evt) {
    if (this.isGrabbed && evt.detail.position) {
      this.el.setAttribute('position', evt.detail.position);
    }
  },

  onHoverStart: function (evt) {
    if (!this.isGrabbed) {
      this.el.setAttribute('material', 'color', this.hoverColor);
    }
  },

  onHoverEnd: function (evt) {
    if (!this.isGrabbed) {
      this.el.setAttribute('material', 'color', this.defaultColor);
    }
  },

  onTriggerDown: function (evt) {
    console.log('Trigger down on button!');
    this.isGrabbed = true;
    this.grabbingController = evt.target.id;
    this.el.setAttribute('material', 'color', this.grabbedColor);
  },

  onTriggerUp: function (evt) {
    console.log('Trigger up on button!');
    this.isGrabbed = false;
    this.grabbingController = null;
    this.el.setAttribute('material', 'color', this.defaultColor);
    this.executeAction();
    this.animatePress();
  },

  onVRGrabStart: function (evt) {
    console.log('VR Grab Start on button!');
    this.isGrabbed = true;
    this.el.setAttribute('material', 'color', this.grabbedColor);
  },

  onVRGrabEnd: function (evt) {
    console.log('VR Grab End on button!');
    this.isGrabbed = false;
    this.el.setAttribute('material', 'color', this.defaultColor);
    this.executeAction();
    this.animatePress();
  },

  onVRClick: function (evt) {
    console.log('VR Click on button!');
    if (!this.isGrabbed) {
      this.executeAction();
      this.animatePress();
    }
  },

  onClick: function (evt) {
    if (!this.isGrabbed) {
      this.executeAction();
      this.animatePress();
    }
  },

  onMouseDown: function (evt) {
    this.isGrabbed = true;
    this.el.setAttribute('material', 'color', this.grabbedColor);
  },

  onMouseUp: function (evt) {
    this.isGrabbed = false;
    this.el.setAttribute('material', 'color', this.defaultColor);
  },

  onGripDown: function (evt) {
    this.isGrabbed = true;
    this.grabbingHand = evt.detail.hand;
    this.el.setAttribute('material', 'color', this.grabbedColor);
  },

  onGripUp: function (evt) {
    this.isGrabbed = false;
    this.grabbingHand = null;
    this.el.setAttribute('material', 'color', this.defaultColor);
  },

  animatePress: function () {
    this.isPressed = true;
    this.el.setAttribute('material', 'color', this.pressedColor);
    this.el.setAttribute('animation', {
      property: 'scale',
      to: '0.9 0.9 0.9',
      dur: 100,
      dir: 'alternate',
      loop: 1
    });

    setTimeout(() => {
      this.isPressed = false;
      this.el.setAttribute('material', 'color', this.defaultColor);
    }, 200);
  },

  executeAction: function () {
    const action = this.data.action;
    const url = this.data.url;
    const label = this.data.label;

    console.log(`Executando ação: ${action} (${label})`);

    this.showNotification(`Executando: ${label}`);

    switch (action) {
      case 'takeoff':
        this.sendDroneCommand('takeoff');
        break;
      case 'land':
        this.sendDroneCommand('land');
        break;
      case 'battery':
        this.sendDroneCommand('battery');
        break;
      case 'flip':
        this.sendDroneCommand('flip', { direction: 'f' });
        break;
      default:
        console.log('Ação não reconhecida:', action);
    }
  },

  sendDroneCommand: function (command, data = {}) {
    if (typeof io !== 'undefined') {
      const socket = io(this.data.url);
      socket.emit(command, data);
      socket.on('response', (response) => {
        console.log('Resposta do drone:', response);
        this.showNotification(`${command}: ${response.status}`);
      });
    } else {
      fetch(`${this.data.url}/api/${command}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      })
        .then(response => response.json())
        .then(data => {
          this.showNotification(`${command}: ${data.status || 'OK'}`);
        })
        .catch(error => {
          this.showNotification(`Erro: ${error.message}`);
        });
    }
  },

  showNotification: function (message) {
    const notification = document.createElement('a-entity');
    const buttonPos = this.el.getAttribute('position');

    notification.setAttribute('position', {
      x: buttonPos.x,
      y: buttonPos.y + 0.5,
      z: buttonPos.z
    });
    notification.setAttribute('text', {
      value: message,
      align: 'center',
      color: '#ffffff',
      width: 6
    });
    notification.setAttribute('geometry', {
      primitive: 'plane',
      width: 2,
      height: 0.3
    });
    notification.setAttribute('material', {
      color: '#333333',
      opacity: 0.8,
      transparent: true
    });

    this.el.sceneEl.appendChild(notification);

    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 3000);
  },

  tick: function () {
    if (this.isGrabbed && this.grabbingHand) {
      const handEl = document.querySelector(`#${this.grabbingHand}Hand`);
      if (handEl) {
        const handPosition = handEl.getAttribute('position');
        if (handPosition) {
          this.el.setAttribute('position', {
            z: handPosition.z - 0.5
          });
        }
      }
    }
  }
});

// VR Debug Component
AFRAME.registerComponent('vr-debug', {
  init: function () {
    console.log('DEBUG: vr-debug component initialized on', this.el.id);
    
    this.el.addEventListener('triggerdown', function(evt) {
      console.log('DEBUG: triggerdown event detected on', evt.target.id);
    });
    
    this.el.addEventListener('triggerup', function(evt) {
      console.log('DEBUG: triggerup event detected on', evt.target.id);
    });

    this.el.addEventListener('gripdown', function(evt) {
      console.log('DEBUG: gripdown event detected on', evt.target.id);
    });

    this.el.addEventListener('gripup', function(evt) {
      console.log('DEBUG: gripup event detected on', evt.target.id);
    });

    this.el.addEventListener('raycaster-intersection', function(evt) {
      console.log('DEBUG: laser intersection with', evt.target.id || 'unnamed object');
    });

    this.el.addEventListener('raycaster-intersection-cleared', function(evt) {
      console.log('DEBUG: laser intersection cleared with', evt.target.id || 'unnamed object');
    });
  }
});

// Simple VR Interaction Component
AFRAME.registerComponent('simple-vr-interaction', {
  init: function () {
    this.isHovered = false;
    this.isGrabbed = false;
    
    this.el.addEventListener('mouseenter', this.onMouseEnter.bind(this));
    this.el.addEventListener('mouseleave', this.onMouseLeave.bind(this));
    this.el.addEventListener('click', this.onClick.bind(this));
    
    this.el.addEventListener('triggerdown', this.onTriggerDown.bind(this));
    this.el.addEventListener('triggerup', this.onTriggerUp.bind(this));
  },

  onMouseEnter: function(evt) {
    console.log('Mouse enter (laser hover) on', this.el.id);
    this.isHovered = true;
    this.el.emit('hover-start');
  },

  onMouseLeave: function(evt) {
    console.log('Mouse leave (laser unhover) on', this.el.id);
    this.isHovered = false;
    this.el.emit('hover-end');
  },

  onClick: function(evt) {
    console.log('Click on', this.el.id);
    this.el.emit('vr-click');
  },

  onTriggerDown: function(evt) {
    console.log('Trigger down on', this.el.id);
    this.isGrabbed = true;
    this.el.emit('vr-grab-start');
  },

  onTriggerUp: function(evt) {
    console.log('Trigger up on', this.el.id);
    this.isGrabbed = false;
    this.el.emit('vr-grab-end');
  }
});
