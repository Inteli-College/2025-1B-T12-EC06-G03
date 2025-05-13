import 'dart:async';
import 'package:dio/dio.dart';
import 'package:socket_io_client/socket_io_client.dart' as IO;
import '../models/drone_command.dart';
import '../models/drone_response.dart';

class DroneService {
  // Default server settings
  static const String DEFAULT_HOST = '10.32.0.11';
  static const int DEFAULT_PORT = 5000;
  
  late String _host;
  late int _port;
  late String _baseUrl;
  late Dio _dio;
  IO.Socket? _socket;
  
  final StreamController<DroneResponse> _responseStreamController = 
      StreamController<DroneResponse>.broadcast();
  
  Stream<DroneResponse> get responseStream => _responseStreamController.stream;
  
  bool _isConnected = false;
  bool get isConnected => _isConnected;
  
  String get host => _host;
  int get port => _port;
  String get baseUrl => _baseUrl;
  
  DroneService({
    String host = DEFAULT_HOST,
    int port = DEFAULT_PORT,
  }) {
    updateServerConfig(host: host, port: port);
  }

  void updateServerConfig({required String host, required int port}) {
    _host = host;
    _port = port;
    _baseUrl = 'http://$_host:$_port';
    _dio = Dio(BaseOptions(baseUrl: _baseUrl));
    
    // Disconnect from current socket if exists
    if (_socket != null && _isConnected) {
      _socket!.disconnect();
      _socket!.dispose();
      _socket = null;
    }
    
    // Initialize socket with new config
    _initSocket();
  }

  void _initSocket() {
    _socket = IO.io(_baseUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': true,
      'reconnection': true,
      'reconnectionAttempts': 10,
      'reconnectionDelay': 1000,
      'reconnectionDelayMax': 5000,
      'timeout': 20000,
    });

    _socket!.onConnect((_) {
      _isConnected = true;
      _responseStreamController.add(
        DroneResponse(
          command: 'connect',
          status: 'ok',
          message: 'Connected to Tello server at $_baseUrl',
        ),
      );
    });

    _socket!.onDisconnect((_) {
      _isConnected = false;
      _responseStreamController.add(
        DroneResponse(
          command: 'disconnect',
          status: 'error',
          message: 'Disconnected from Tello server',
        ),
      );
    });
    
    // Adicionar handler para erros de conexão
    _socket!.onError((error) {
      print('Socket connection error: $error');
      _responseStreamController.add(
        DroneResponse(
          command: 'error',
          status: 'error',
          message: 'Connection error: $error',
        ),
      );
    });
    
    // Handler para tentativos de reconexão
    _socket!.onReconnect((_) {
      print('Attempting to reconnect to server...');
    });
    
    // Handler para reconexão bem sucedida
    _socket!.onReconnect((_) {
      print('Successfully reconnected to server');
      _isConnected = true;
      _responseStreamController.add(
        DroneResponse(
          command: 'connect',
          status: 'ok',
          message: 'Reconnected to Tello server',
        ),
      );
    });

    // Configurar handlers para diferentes tipos de eventos
    _socket!.on('response', (data) {
      print('Received generic response: $data');
      _responseStreamController.add(DroneResponse.fromJson(data));
    });
    
    // Handler específico para respostas de bateria
    _socket!.on('battery', (data) {
      print('Received battery response: $data');
      
      // Processar dados específicos da bateria - tratando diferentes formatos possíveis
      if (data is Map) {
        // Se for um mapa, pode ter campos específicos
        _responseStreamController.add(DroneResponse(
          command: 'battery',
          status: 'ok',
          response: data['response'] ?? data['value'],
          message: data['message']?.toString() ?? data.toString(),
        ));
      } else if (data is int || (data is String && int.tryParse(data) != null)) {
        // Se for diretamente o valor da bateria como int ou string numérica
        final int batteryValue = data is int ? data : int.parse(data);
        _responseStreamController.add(DroneResponse(
          command: 'battery',
          status: 'ok',
          response: batteryValue,
          message: batteryValue.toString(),
        ));
      } else {
        // Caso seja outro formato, tenta extrair o máximo de informação
        _responseStreamController.add(DroneResponse(
          command: 'battery',
          status: 'ok',
          response: data,
          message: data.toString(),
        ));
      }
    });

    _socket!.connect();
  }

  Future<void> sendCommand(DroneCommand command) async {
    if (_socket == null || !_isConnected) {
      _responseStreamController.add(
        DroneResponse(
          command: command.command,
          status: 'error',
          message: 'Not connected to server',
        ),
      );
      return;
    }

    // Special handling for specific commands
    if (command.command == 'flip') {
      final flipData = command.toJson();
      // Log for debugging
      print('Sending flip command with direction: ${flipData['direction']}');
      
      try {
        // Ensure we're sending the flip command exactly as the backend expects it
        _socket!.emit('flip', {'direction': flipData['direction']});
        print('Emitted flip event with data: ${{'direction': flipData['direction']}}');
      } catch (e) {
        print('Error sending flip command: $e');
        _responseStreamController.add(
          DroneResponse(
            command: command.command,
            status: 'error',
            message: 'Error sending flip command: $e',
          ),
        );
      }
    } 
    else if (command.command == 'battery') {
      // Tratamento específico para o comando de bateria
      print('Sending battery command');
      try {
        _socket!.emit('battery');
        
        // Adicionar log para depuração
        print('Battery command sent successfully');
      } catch (e) {
        print('Error sending battery command: $e');
        _responseStreamController.add(
          DroneResponse(
            command: command.command,
            status: 'error',
            message: 'Error sending battery command: $e',
          ),
        );
      }
    } 
    else {
      // Standard handling for other commands
      _socket!.emit(command.command, command.toJson());
    }
  }

  // Novo método para enviar controle RC
  Future<void> sendRCControl(Map<String, int> params) async {
    if (_socket == null || !_isConnected) {
      _responseStreamController.add(
        DroneResponse(
          command: 'rc_control',
          status: 'error',
          message: 'Not connected to server',
        ),
      );
      return;
    }

    try {
      // Validar os parâmetros
      final lr = params['lr'] ?? 0;   // left/right
      final fb = params['fb'] ?? 0;   // forward/backward
      final ud = params['ud'] ?? 0;   // up/down
      final yw = params['yw'] ?? 0;   // yaw (rotação)

      // Garantir que os valores estejam dentro da faixa -100 a 100
      final validParams = {
        'lr': _clampValue(lr, -100, 100),
        'fb': _clampValue(fb, -100, 100),
        'ud': _clampValue(ud, -100, 100),
        'yw': _clampValue(yw, -100, 100),
      };
      
      // Log para depuração
      print('Sending RC control: lr=$lr, fb=$fb, ud=$ud, yw=$yw');
      
      // Emitir evento rc_control
      _socket!.emit('rc_control', validParams);
    } catch (e) {
      print('Error sending RC control: $e');
      _responseStreamController.add(
        DroneResponse(
          command: 'rc_control',
          status: 'error',
          message: 'Error sending RC control: $e',
        ),
      );
    }
  }

  // Utilitário para limitar valores
  int _clampValue(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }

  String getVideoStreamUrl() {
    return '$_baseUrl/video';
  }

  void disconnect() {
    if (_socket != null) {
      _socket!.disconnect();
    }
  }

  void dispose() {
    if (_socket != null) {
      _socket!.dispose();
    }
    _responseStreamController.close();
  }
}