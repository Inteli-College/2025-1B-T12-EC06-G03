---
title: Componentes A-Frame Customizados
sidebar_position: 2
---

# Componentes A-Frame Customizados

Esta documentação detalha os componentes personalizados desenvolvidos para a aplicação VR de controle de drone.

## Visão Geral

A aplicação utiliza componentes A-Frame customizados para implementar funcionalidades específicas de interação VR e controle de drone. Cada componente é responsável por uma funcionalidade específica e pode ser reutilizado em diferentes entidades.

## moveable-screen

Componente responsável por tornar a tela de vídeo interativa e movível no espaço VR.

### Schema

```javascript
schema: {
  moveSpeed: { type: 'number', default: 0.1 },
  rotateSpeed: { type: 'number', default: 1.0 },
  scaleSpeed: { type: 'number', default: 0.1 },
  minScale: { type: 'number', default: 0.5 },
  maxScale: { type: 'number', default: 3.0 }
}
```

### Funcionalidades

- **Movimento**: Arrastar e soltar no espaço 3D
- **Rotação**: Controle de orientação da tela
- **Escala**: Redimensionamento dinâmico
- **Limites**: Restrições de tamanho mínimo e máximo

### Eventos Suportados

| Evento | Descrição | Trigger |
|--------|-----------|---------|
| `onGrabStart` | Início de interação | Controlador VR ou hand tracking |
| `onGrabEnd` | Fim de interação | Soltar objeto |
| `onDragMove` | Movimento durante drag | Movimento contínuo |
| `onHoverStart` | Início de hover | Ponteiro sobre objeto |
| `onHoverEnd` | Fim de hover | Ponteiro sai do objeto |

### Uso

```html
<a-entity moveable-screen="moveSpeed: 0.2; scaleSpeed: 0.15">
  <a-plane material="src: #videoStream"></a-plane>
</a-entity>
```

### Estados Visuais

- **Normal**: Cor padrão branca (`#ffffff`)
- **Hover**: Sem mudança visual (mantém cor)
- **Grabbed**: Verde claro (`#88ff88`)

## interactive-button

Componente para criar botões interativos que executam comandos de drone.

### Schema

```javascript
schema: {
  label: { type: 'string', default: 'Button' },
  action: { type: 'string', default: '' },
  url: { type: 'string', default: '' }
}
```

### Ações Disponíveis

| Ação | Comando | Descrição |
|------|---------|-----------|
| `takeoff` | POST /takeoff | Comando de decolagem |
| `land` | POST /land | Comando de pouso |
| `battery` | GET /battery | Verificar nível de bateria |
| `flip` | POST /flip | Executar manobra flip |

### Eventos Suportados

```javascript
// Eventos VR
onVRGrabStart()    // Início de grab com controlador
onVRGrabEnd()      // Fim de grab com controlador
onTriggerDown()    // Pressionar trigger
onTriggerUp()      // Soltar trigger

// Eventos Mouse (Desktop)
onMouseDown()      // Pressionar botão mouse
onMouseUp()        // Soltar botão mouse

// Eventos Específicos
onGripDown()       // Pressionar grip controller
onGripUp()         // Soltar grip controller
```

### Estados Visuais

- **Default**: Cor específica por tipo de botão
- **Hover**: Iluminação aumentada
- **Pressed**: Cor de destaque (`#grabbedColor`)
- **Active**: Animação de pressão

### Uso

```html
<a-entity interactive-button="action: takeoff; label: TAKEOFF; url: http://10.140.0.11:5000">
  <a-cylinder radius="0.2" height="0.1" color="#4CAF50"></a-cylinder>
</a-entity>
```

### Feedback Visual

```javascript
animatePress: function () {
  // Animação de escala ao pressionar
  this.el.setAttribute('animation', {
    property: 'scale',
    to: '0.9 0.9 0.9',
    dur: 100,
    easing: 'easeInOutQuad'
  });
  
  // Retorno ao tamanho original
  setTimeout(() => {
    this.el.setAttribute('animation', {
      property: 'scale',
      to: '1 1 1',
      dur: 100
    });
  }, 200);
}
```

## keyboard-controls

Componente para controle via teclado (fallback para desktop).

### Teclas Suportadas

| Tecla | Ação | Objeto |
|-------|------|--------|
| `WASD` | Movimento | Tela de vídeo |
| `QE` | Rotação Y | Tela de vídeo |
| `RF` | Rotação X | Tela de vídeo |
| `ZX` | Escala | Tela de vídeo |
| `1-4` | Comandos drone | Botões específicos |

### Implementação

```javascript
bindEvents: function () {
  document.addEventListener('keydown', (evt) => {
    switch(evt.key.toLowerCase()) {
      case 'w': this.moveScreen(0, 0.1, 0); break;
      case 's': this.moveScreen(0, -0.1, 0); break;
      case 'a': this.moveScreen(-0.1, 0, 0); break;
      case 'd': this.moveScreen(0.1, 0, 0); break;
      // ... mais controles
    }
  });
}
```

### Uso

```html
<a-entity keyboard-controls></a-entity>
```

## vr-debug

Componente de debug para desenvolvimento e diagnóstico.

### Funcionalidades

- **Logging de Eventos**: Registra todas as interações VR
- **Estado dos Controladores**: Monitora conexão e status
- **Performance Metrics**: FPS e timing de operações
- **Diagnóstico de Problemas**: Detecção automática de issues

### Logs Gerados

```javascript
init: function () {
  console.log('[VR-DEBUG] Component initialized');
  this.logControllerStatus();
  this.setupEventListeners();
}

logControllerStatus: function () {
  console.log('[VR-DEBUG] Left Controller:', this.leftController?.connected);
  console.log('[VR-DEBUG] Right Controller:', this.rightController?.connected);
  console.log('[VR-DEBUG] Hand Tracking:', this.handTracking?.active);
}
```

### Eventos Monitorados

- Conexão/desconexão de controladores
- Eventos de interação (grab, release, hover)
- Erros de WebXR
- Performance warnings

### Uso

```html
<a-entity vr-debug></a-entity>
```

## simple-vr-interaction

Componente simplificado para interações básicas VR.

### Eventos Mapeados

```javascript
init: function () {
  // Mapear eventos nativos para eventos customizados
  this.el.addEventListener('triggerup', this.onTriggerUp.bind(this));
  this.el.addEventListener('mouseenter', this.onMouseEnter.bind(this));
  this.el.addEventListener('mouseleave', this.onMouseLeave.bind(this));
  this.el.addEventListener('click', this.onClick.bind(this));
}
```

### Eventos Emitidos

- `hover-start`: Quando ponteiro entra no objeto
- `hover-end`: Quando ponteiro sai do objeto  
- `vr-click`: Clique VR ou mouse
- `vr-grab-start`: Início de grab
- `vr-grab-end`: Fim de grab

### Uso

```html
<a-entity simple-vr-interaction class="interactive">
  <!-- Conteúdo do objeto -->
</a-entity>
```

## Boas Práticas de Uso

### 1. Classes CSS

Sempre adicionar classes apropriadas para seletores:

```html
<a-entity class="interactive grabbable hoverable clickable">
```

### 2. Configuração de Raycaster

Configurar raycaster para detectar objetos interativos:

```html
<a-entity raycaster="objects: .interactive; showLine: true; far: 20">
```

### 3. Hierarquia de Componentes

Ordem recomendada de componentes:

```html
<a-entity 
  moveable-screen
  keyboard-controls
  grabbable draggable droppable hoverable clickable
  class="interactive grabbable"
  simple-vr-interaction
  vr-debug>
```

### 4. Performance

- Use `tick()` com moderação
- Implemente debouncing para eventos frequentes
- Cache referências de elementos

### 5. Debug

Sempre incluir `vr-debug` durante desenvolvimento:

```html
<a-entity vr-debug id="debug-info"></a-entity>
```

## Extensibilidade

### Criando Novos Componentes

Template básico:

```javascript
AFRAME.registerComponent('meu-componente', {
  schema: {
    propriedade: { type: 'string', default: 'valor' }
  },
  
  init: function () {
    // Inicialização
    this.setupEventListeners();
  },
  
  setupEventListeners: function () {
    // Configurar eventos
  },
  
  tick: function (time, timeDelta) {
    // Loop de atualização (use com cuidado)
  },
  
  remove: function () {
    // Limpeza ao remover componente
  }
});
```

### Integração com Sistema Existente

Para integrar novos componentes:

1. Registrar o componente antes do `<a-scene>`
2. Adicionar à classe `.interactive` se aplicável
3. Implementar eventos padrão (`onGrabStart`, `onHoverStart`, etc.)
4. Incluir debug logging
5. Documentar schema e eventos

## Troubleshooting

### Componente não funciona

1. Verificar se está registrado antes do `<a-scene>`
2. Confirmar sintaxe do schema
3. Verificar console para erros JavaScript

### Eventos não disparam

1. Verificar se elemento tem classe `.interactive`
2. Confirmar configuração do raycaster
3. Testar com mouse primeiro (desktop)

### Performance ruim

1. Revisar uso do `tick()`
2. Implementar throttling/debouncing
3. Verificar vazamentos de memória em event listeners
