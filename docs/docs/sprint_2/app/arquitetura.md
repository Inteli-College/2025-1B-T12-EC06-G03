---
title: Arquitetura Técnica
sidebar_position: 4
---

# Arquitetura Técnica

## Camada de Apresentação (UI)

Esta camada contém todos os widgets Flutter que compõem a interface do usuário:

- **Pages**: Telas completas do aplicativo
  - `HomePage`: Tela principal com controles do drone
  - `ServerConfigPage`: Tela de configuração de conexão
  
- **Widgets**: Componentes reutilizáveis
  - `DroneControlPanel`: Painel com botões de comando principais
  - `DroneJoystickControl`: Controle virtual para movimentação horizontal
  - `AltitudeJoystickControl`: Controle vertical e rotacional
  - `VideoStreamWidget`: Componente para exibição do streaming de vídeo
  - `StatusNotification`: Alertas e notificações do sistema
  - `SplashScreen`: Tela de inicialização

## Camada de Gerenciamento de Estado (BLoC)

Implementa o padrão BLoC (Business Logic Component) para gerenciar o fluxo de dados e estado do aplicativo:

- `DroneBloc`: Gerencia o estado geral do sistema de controle do drone
  ```dart
  class DroneBloc extends Bloc<DroneEvent, DroneState> {
    final DroneService _droneService;
    StreamSubscription? _droneResponseSubscription;
    
    DroneBloc(this._droneService) : super(DroneState.initial()) {
      // Registra handlers para diferentes eventos
      on<ConnectToDrone>(_onConnectToDrone);
      on<DisconnectFromDrone>(_onDisconnectFromDrone);
      on<SendDroneCommand>(_onSendDroneCommand);
      // ...
    }
    
    // Handlers para eventos específicos
    Future<void> _onConnectToDrone(ConnectToDrone event, Emitter<DroneState> emit) async {
      // Implementação de conexão
    }
    
    // ...
  }
  ```

- `DroneEvent`: Classes que representam eventos no sistema
  ```dart
  abstract class DroneEvent {}
  
  class ConnectToDrone extends DroneEvent {
    final String host;
    final int port;
    
    ConnectToDrone({required this.host, required this.port});
  }
  
  class SendDroneCommand extends DroneEvent {
    final DroneCommand command;
    
    SendDroneCommand(this.command);
  }
  
  // ...
  ```

- `DroneState`: Representa o estado atual do sistema
  ```dart
  class DroneState {
    final ConnectionStatus connectionStatus;
    final ServerConfig serverConfig;
    final bool isExecutingCommand;
    final String? lastMessage;
    final MessageType lastMessageType;
    final bool showNotification;
    final Map<String, dynamic> telemetryData;
    
    // ...
  }
  ```

## Camada de Serviços

Responsável pela comunicação com sistemas externos:

- `DroneService`: Gerencia a comunicação com o servidor de controle do drone
  ```dart
  class DroneService {
    late String _host;
    late int _port;
    late String _baseUrl;
    late Dio _dio;
    IO.Socket? _socket;
    
    final StreamController<DroneResponse> _responseStreamController;
    
    // Métodos de comunicação
    Future<void> connect(String host, int port) async {
      // Implementação da conexão
    }
    
    Future<DroneResponse> sendCommand(DroneCommand command) async {
      // Envio de comandos
    }
    
    // ...
  }
  ```

## Camada de Modelos

Define as estruturas de dados utilizadas no aplicativo:

- `DroneCommand`: Representa um comando a ser enviado ao drone
  ```dart
  class DroneCommand {
    final String action;
    final Map<String, dynamic> parameters;
    
    const DroneCommand({
      required this.action,
      this.parameters = const {},
    });
    
    // Comandos predefinidos
    factory DroneCommand.takeoff() => const DroneCommand(action: 'takeoff');
    factory DroneCommand.land() => const DroneCommand(action: 'land');
    // ...
    
    Map<String, dynamic> toJson() => {
      'action': action,
      'parameters': parameters,
    };
  }
  ```

- `DroneResponse`: Encapsula a resposta do servidor do drone
  ```dart
  class DroneResponse {
    final bool success;
    final String message;
    final Map<String, dynamic>? data;
    
    const DroneResponse({
      required this.success,
      required this.message,
      this.data,
    });
    
    // ...
  }
  ```

- `ServerConfig`: Armazena configurações de conexão com o servidor
  ```dart
  class ServerConfig {
    final String host;
    final int port;
    final String? savePath;
    
    const ServerConfig({
      required this.host,
      required this.port,
      this.savePath,
    });
    
    // ...
  }
  ```

## Fluxo de Dados

O fluxo de dados no aplicativo segue um padrão unidirecional:

1. **Evento de Interação do Usuário**: Um evento é disparado pela interação do usuário com a interface
2. **Processamento pelo BLoC**: O `DroneBloc` processa o evento e chama o `DroneService` conforme necessário
3. **Comunicação com o Servidor**: O `DroneService` envia comandos ao servidor e aguarda resposta
4. **Atualização do Estado**: Com base na resposta, o `DroneBloc` atualiza o `DroneState`
5. **Renderização da UI**: Os widgets reagem à mudança de estado e atualizam a interface

## Comunicação com o Servidor

O aplicativo utiliza dois métodos principais para comunicação com o servidor:

### Socket.IO para Comunicação em Tempo Real

Utilizado para:
- Streaming de telemetria
- Comandos de controle RC em tempo real
- Notificações e alertas em tempo real

```dart
void _setupSocketListeners() {
  _socket?.on('battery', (data) {
    final response = DroneResponse(
      success: true,
      message: 'Battery data received',
      data: data,
    );
    _responseStreamController.add(response);
  });
  
  // ...
}
```

### HTTP (Dio) para Comandos Específicos

Utilizado para:
- Comandos que exigem confirmação (decolagem, pouso, RTH)
- Upload/download de arquivos
- Configurações e atualizações do sistema

```dart
Future<DroneResponse> _sendHttpCommand(DroneCommand command) async {
  try {
    final response = await _dio.post(
      '/command',
      data: command.toJson(),
      options: Options(
        sendTimeout: const Duration(seconds: 5),
        receiveTimeout: const Duration(seconds: 5),
      ),
    );
    
    return DroneResponse(
      success: response.data['success'] ?? false,
      message: response.data['message'] ?? 'No message',
      data: response.data['data'],
    );
  } catch (e) {
    // Tratamento de erros
    return DroneResponse(
      success: false,
      message: 'Error: ${e.toString()}',
    );
  }
}
```

## Gerenciamento de Recursos

### Gerenciamento de Memória

O aplicativo implementa estratégias eficientes de gerenciamento de memória:

- Cancelamento automático de `StreamSubscription` quando não mais necessário
- Liberação de recursos de vídeo quando o streaming é interrompido
- Uso eficiente de imagens e assets com cache apropriado

### Gerenciamento de Energia

Para otimizar o uso da bateria do dispositivo móvel:

- Ajuste dinâmico da taxa de atualização da UI com base na atividade
- Redução da frequência de comunicação quando o drone está em modo estacionário
- Otimização da decodificação de vídeo para reduzir o processamento
