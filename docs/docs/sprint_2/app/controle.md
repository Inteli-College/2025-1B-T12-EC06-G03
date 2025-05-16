---
title: Sistema de Controle
sidebar_position: 1
---

# Sistema de Controle

## Arquitetura de Controle

O sistema de controle do aplicativo Athenas é baseado em uma arquitetura cliente-servidor, onde o aplicativo Flutter atua como cliente enviando comandos para o servidor que está conectado diretamente ao drone. Esta seção descreve os componentes principais do sistema de controle.

### Componentes Principais

1. **Controles de Interface do Usuário**
   - Joysticks virtuais (direcional e de altitude)
   - Botões de comando rápido (decolagem, pouso, retorno, parada de emergência)
   - Controles de câmera (captura de foto, início/parada de gravação)
   - Indicadores de status e telemetria

2. **Gerenciamento de Estado com BLoC**
   - `DroneBloc`: Coordena o fluxo de dados entre a UI e os serviços
   - `DroneEvent`: Eventos desencadeados pela interação do usuário ou respostas do sistema
   - `DroneState`: Representa o estado atual do sistema de controle e do drone

3. **Serviço de Comunicação**
   - `DroneService`: Gerencia a comunicação com o servidor do drone
   - Utiliza Socket.io para comunicação em tempo real
   - Implementa requisições HTTP para comandos específicos

## Fluxo de Controle

O fluxo de controle segue o seguinte padrão:

1. O usuário interage com um controle na interface
2. A interação gera um `DroneEvent`
3. O `DroneBloc` processa o evento e chama os métodos apropriados no `DroneService`
4. O `DroneService` envia o comando ao servidor através de Socket.io ou HTTP
5. O servidor processa o comando e o transmite ao drone
6. O drone executa o comando e retorna informações de status
7. As informações de status são recebidas pelo servidor e transmitidas de volta ao aplicativo
8. O `DroneService` converte a resposta em um objeto `DroneResponse`
9. O `DroneBloc` atualiza o `DroneState` com base na resposta
10. A UI é atualizada para refletir o novo estado

## Tipos de Comandos

### Comandos Básicos

| Comando | Descrição | Implementação |
|---------|-----------|---------------|
| Takeoff | Decolar o drone | `DroneCommand.takeoff()` |
| Land | Pousar o drone | `DroneCommand.land()` |
| RTH | Retorno ao ponto inicial | `DroneCommand.returnToHome()` |
| Emergency Stop | Parada de emergência | `DroneCommand.emergencyStop()` |

### Controles RC (Remote Control)

Os controles RC permitem o movimento preciso do drone em todos os eixos:

```dart
RCControlCommand(
  roll: _lr,       // Movimento lateral (-1.0 a 1.0)
  pitch: _fb,      // Movimento frente/trás (-1.0 a 1.0)
  throttle: _ud,   // Altitude (-1.0 a 1.0)
  yaw: _yw,        // Rotação (-1.0 a 1.0)
)
```

## Implementação dos Joysticks

O aplicativo utiliza dois joysticks virtuais para controle preciso do drone:

### Joystick Direcional (DroneJoystickControl)

Este joystick controla os movimentos horizontais do drone:
- Eixo X: Movimento lateral (roll)
- Eixo Y: Movimento frente/trás (pitch)

```dart
JoystickView(
  size: widget.size,
  backgroundColor: Colors.blueGrey.withOpacity(0.2),
  innerCircleColor: Theme.of(context).colorScheme.primary,
  outerCircleColor: Colors.grey.withOpacity(0.5),
  onDirectionChanged: _onDirectionChanged,
)
```

### Joystick de Altitude (AltitudeJoystickControl)

Este joystick controla:
- Eixo Y: Subida/descida (throttle)
- Eixo X: Rotação (yaw)

## Sistema de Feedback

### Feedback Visual

- **Indicadores de Estado**: Mostram o status atual da conexão e do drone
- **Feedback de Telemetria**: Exibem informações como altitude, velocidade e nível de bateria
- **Streaming de Vídeo**: Fornece feedback visual em tempo real da câmera do drone

### Feedback Tátil

Em dispositivos que suportam, o aplicativo fornece feedback haptico para:
- Confirmação de comandos
- Alertas de bateria baixa
- Avisos de perda de conexão

## Tratamento de Erros

O sistema de controle implementa mecanismos robustos de tratamento de erros:

- **Perda de Conexão**: Detecção automática e tentativas de reconexão
- **Timeout de Comandos**: Monitoramento do tempo de resposta e cancelamento automático de comandos pendentes
- **Limites de Segurança**: Restrições de movimento baseadas no estado da bateria e na qualidade do sinal

## Simulação e Testes

Para facilitar testes sem um drone físico, o aplicativo inclui um modo de simulação onde:
- Os comandos são processados localmente
- Os estados do drone são simulados
- O feedback visual é gerado sinteticamente

Este modo é crucial para desenvolvimento e testes de novas funcionalidades antes da implantação em drones reais.
