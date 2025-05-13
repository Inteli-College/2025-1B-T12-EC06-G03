import 'package:equatable/equatable.dart';
import '../models/server_config.dart';

enum ConnectionStatus { disconnected, connecting, connected }

class DroneState extends Equatable {
  final ConnectionStatus connectionStatus;
  final int? batteryLevel;
  final String? lastCommandStatus;
  final String? lastCommandMessage;
  final bool isExecutingCommand;
  final ServerConfig serverConfig;
  final String? lastCommand;

  const DroneState({
    this.connectionStatus = ConnectionStatus.disconnected,
    this.batteryLevel,
    this.lastCommandStatus,
    this.lastCommandMessage,
    this.isExecutingCommand = false,
    this.serverConfig = const ServerConfig(host: '10.32.0.11', port: 5000),
    this.lastCommand,
  });
  
  // Static method to create the initial state
  static DroneState initial() {
    return const DroneState();
  }
  
  // Getter for connection status
  bool get isConnected => connectionStatus == ConnectionStatus.connected;

  DroneState copyWith({
    ConnectionStatus? connectionStatus,
    int? batteryLevel,
    String? lastCommandStatus,
    String? lastCommandMessage,
    bool? isExecutingCommand,
    ServerConfig? serverConfig,
    String? lastCommand,
    bool? isConnecting,
  }) {
    return DroneState(
      connectionStatus: connectionStatus ?? this.connectionStatus,
      batteryLevel: batteryLevel ?? this.batteryLevel,
      lastCommandStatus: lastCommandStatus ?? this.lastCommandStatus,
      lastCommandMessage: lastCommandMessage ?? this.lastCommandMessage,
      isExecutingCommand: isExecutingCommand ?? this.isExecutingCommand,
      serverConfig: serverConfig ?? this.serverConfig,
      lastCommand: lastCommand ?? this.lastCommand,
    );
  }

  @override
  List<Object?> get props => [
    connectionStatus,
    batteryLevel,
    lastCommandStatus,
    lastCommandMessage,
    isExecutingCommand,
    serverConfig,
    lastCommand,
  ];
}