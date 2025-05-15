---
title: Sistema de Comunicação
sidebar_position: 5
---

# Sistema de Comunicação

## Visão Geral

O sistema de comunicação do aplicativo Athenas é responsável por estabelecer e manter a conexão entre o app Flutter e o servidor de controle do drone. A arquitetura de comunicação foi projetada para ser robusta, eficiente e capaz de lidar com conexões de baixa qualidade, comuns em operações de campo.

## Protocolos Utilizados

O app utiliza dois protocolos principais para comunicação:

### 1. Socket.IO

Socket.IO é utilizado para comunicação em tempo real e bidirecional entre o aplicativo e o servidor. Este protocolo é especialmente adequado para:

- **Controles RC**: Comandos de joystick que precisam ser enviados com alta frequência
- **Bateria**: Atualizações em tempo real sobre o nível de bateria do drone
- **Alertas**: Notificações urgentes do drone para o aplicativo

```dart
void _setupSocket(String host, int port) {
  final options = IO.OptionBuilder()
    .setTransports(['websocket'])
    .disableAutoConnect()
    .setReconnectionAttempts(10)
    .setReconnectionDelay(3000)
    .setReconnectionDelayMax(5000)
    .build();

  _socket = IO.io('http://$host:$port', options);
  
  _socket?.onConnect((_) {
    _isConnected = true;
    final response = DroneResponse(
      success: true,
      message: 'Connected to drone server',
    );
    _responseStreamController.add(response);
  });
  
  _socket?.onDisconnect((_) {
    _isConnected = false;
    final response = DroneResponse(
      success: false,
      message: 'Disconnected from drone server',
    );
    _responseStreamController.add(response);
  });
  
  _setupSocketListeners();
  _socket?.connect();
}
```

### 2. HTTP REST API

Para comandos menos frequentes e que necessitam de confirmação explícita, o aplicativo utiliza chamadas HTTP REST:

- **Video Streaming**: Transmissão contínua do feed de vídeo da câmera do drone

```dart
Future<DroneResponse> _sendHttpCommand(DroneCommand command) async {
  try {
    final response = await _dio.post(
      '/command',
      data: command.toJson(),
    );
    
    return DroneResponse.fromJson(response.data);
  } catch (e) {
    return DroneResponse(
      success: false,
      message: 'Error sending command: ${e.toString()}',
    );
  }
}
```


## Estratégias de Conexão

### Estabelecimento de Conexão

O processo de conexão com o servidor segue estas etapas:

1. **Handshake Inicial**: Estabelecimento da conexão Socket.IO
2. **Sincronização**: Solicitação do estado atual do drone
3. **Configuração de Streaming**: Definição dos parâmetros de qualidade do streaming de vídeo

### Manutenção da Conexão

Para manter a conexão estável durante operações prolongadas:

- **Heartbeat**: Mensagens periódicas para verificar a conectividade
- **Reconexão Automática**: Tentativas automáticas de reconexão em caso de perda de sinal
- **Exponential Backoff**: Intervalo crescente entre tentativas de reconexão para evitar sobrecarga

```dart
// Configuração de reconnection policy
final options = IO.OptionBuilder()
  .setReconnectionAttempts(10)
  .setReconnectionDelay(3000)  // 3 segundos iniciais
  .setReconnectionDelayMax(30000)  // Máximo de 30 segundos
  .build();
```

## Streaming de Vídeo

O streaming de vídeo da câmera do drone é implementado utilizando:

- **MJPEG Streaming**: Para compatibilidade máxima em diferentes plataformas
- **WebRTC**: Quando disponível, para menor latência e melhor qualidade

```dart
class VideoStreamWidget extends StatelessWidget {
  final String streamUrl;
  final bool isConnected;
  
  const VideoStreamWidget({
    Key? key,
    required this.streamUrl,
    required this.isConnected,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    if (!isConnected) {
      return const NoConnectionPlaceholder();
    }
    
    return MjpegView(
      url: streamUrl,
      fit: BoxFit.contain,
      timeout: const Duration(seconds: 5),
      isLive: true,
    );
  }
}
```

### Adaptação de Qualidade

O sistema ajusta dinamicamente a qualidade do streaming com base em:

- Qualidade da conexão de rede
- Capacidade de processamento do dispositivo
- Nível de bateria do dispositivo e do drone

## Tratamento de Erros

### Detecção de Erros

O sistema está equipado para detectar uma variedade de erros de comunicação:

- **Timeout**: Sem resposta do servidor após um período definido
- **Erros de Socket**: Falhas na conexão WebSocket
- **Erros HTTP**: Respostas de erro do servidor REST
- **Dados Inválidos**: Respostas malformadas ou inesperadas

### Recuperação de Erros

Para cada tipo de erro, estratégias específicas de recuperação são implementadas:

- **Retry Logic**: Repetição automática de comandos críticos
- **Degradação Graciosa**: Redução da qualidade do streaming em vez de interrupção
- **Comandos de Fallback**: Comandos de segurança automáticos em caso de perda prolongada de conexão

```dart
Future<DroneResponse> sendCommand(DroneCommand command, {int retries = 3}) async {
  for (int attempt = 0; attempt < retries; attempt++) {
    try {
      // Tentativa de enviar o comando
      final response = await _sendHttpCommand(command);
      if (response.success) {
        return response;
      }
      
      // Espera crescente entre tentativas
      await Future.delayed(Duration(milliseconds: 500 * (attempt + 1)));
    } catch (e) {
      if (attempt == retries - 1) {
        return DroneResponse(
          success: false,
          message: 'Failed after $retries attempts: ${e.toString()}',
        );
      }
    }
  }
  
  return DroneResponse(
    success: false,
    message: 'Command failed after $retries attempts',
  );
}
```
