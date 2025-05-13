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

    _socket!.on('response', (data) {
      _responseStreamController.add(DroneResponse.fromJson(data));
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

    // Special handling for flip command to ensure it matches backend format
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
    } else {
      // Standard handling for other commands
      _socket!.emit(command.command, command.toJson());
    }
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