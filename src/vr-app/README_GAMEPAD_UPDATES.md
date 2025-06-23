# Atualizações do Sistema de Controle por Joystick/Gamepad

## Resumo das Mudanças

Implementei uma nova lógica de controle por joystick baseada no exemplo fornecido, mantendo a funcionalidade de exibir valores no texto. As principais mudanças incluem:

### Novos Componentes

#### 1. `gamepad-listener` (Novo - Estilo VR)
- Baseado no exemplo fornecido que usa `tracked-controls`
- Acessa o gamepad através de `this.el.components["tracked-controls"]?.controller`
- Ideal para controles VR como Oculus Touch
- Move objetos usando os eixos `[x, y]` do joystick
- Multiplica movimento por velocidade configurável (`speed = 0.05`)

#### 2. `joystick-controller` (Atualizado - Legacy)
- Mantém compatibilidade com gamepads tradicionais (Xbox, PlayStation, etc.)
- Usa `navigator.getGamepads()` para detectar controles
- Implementa a nova lógica de movimento baseada nos eixos `[x, y]`
- Move elementos usando a mesma abordagem do exemplo fornecido

### Funcionalidades Mantidas

✅ **Exibição de Valores no Texto**: Ambos os componentes continuam mostrando:
- Valores dos analógicos formatados
- Botões pressionados
- Status de conexão do controle

✅ **Controle de Câmera**: Analógico direito continua controlando a rotação da câmera

✅ **Sistema de Debug**: Componente `joystick-debug-display` mantido inalterado

## Como Usar

### Para Controles VR (Recomendado)
```html
<a-entity id="right-controller" 
          oculus-touch-controls="hand: right" 
          gamepad-listener>
</a-entity>
```

### Para Gamepads Tradicionais
```html
<a-scene joystick-controller>
  <!-- conteúdo da cena -->
</a-scene>
```

## Elementos Controláveis

Os componentes procuram por elementos nesta ordem de prioridade:
1. `#videoScreen` (tela de vídeo principal)
2. `#cube` (cubo de teste)
3. `a-box` (qualquer caixa na cena)

## Controles

### Joystick VR (`gamepad-listener`)
- **Joystick Principal**: Move o elemento controlado no plano X-Z
- **Botões**: Logados no console e exibidos no texto de status
- **Analógicos Adicionais**: Controlam rotação da câmera (se disponíveis)

### Gamepad Tradicional (`joystick-controller`)
- **Analógico Esquerdo (eixos 0,1)**: Move o elemento controlado no plano X-Z
- **Analógico Direito (eixos 2,3)**: Controla rotação da câmera
- **Botões**: Logados no console e exibidos no texto de status

## Configurações

### Velocidade de Movimento
```javascript
// No gamepad-listener
this.speed = 0.05; // Ajustável no init()

// No joystick-controller
const speed = 0.05; // Ajustável no tick()
```

### Zona Morta
Ambos os componentes usam zona morta de `0.1` para evitar drift dos analógicos.

## Elementos Adicionados na Cena

- **Cubo de Teste**: `<a-box id="cube">` - elemento visual para testar movimento
- **Painel de Instruções Atualizado**: Mostra controles para VR e gamepad tradicional
- **Suporte WebXR**: Configurações atualizadas para melhor compatibilidade VR

## Compatibilidade

- ✅ **Controles VR**: Oculus Touch, HTC Vive, etc.
- ✅ **Gamepads Tradicionais**: Xbox, PlayStation, genéricos
- ✅ **Navegadores**: Chrome, Firefox, Edge (com WebXR)
- ✅ **Dispositivos**: PC, dispositivos móveis compatíveis

## Debug

Para debug detalhado, use o componente existente:
```html
<a-scene joystick-debug-display>
```

Isso criará um painel verde com informações completas do gamepad.
