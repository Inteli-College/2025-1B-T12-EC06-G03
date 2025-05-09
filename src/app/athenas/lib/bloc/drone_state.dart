import 'package:equatable/equatable.dart';
import '../models/drone_response.dart';
import '../models/server_config.dart';

enum ConnectionStatus { disconnected, connecting, connected }

class DroneState extends Equatable {
  final ConnectionStatus connectionStatus;
  final int? batteryLevel;
  final String? lastCommandStatus;
  final String? lastCommandMessage;
  final bool isExecutingCommand;
  final ServerConfig serverConfig;

  const DroneState({
    this.connectionStatus = ConnectionStatus.disconnected,
    this.batteryLevel,
    this.lastCommandStatus,
    this.lastCommandMessage,
    this.isExecutingCommand = false,
    this.serverConfig = const ServerConfig(host: 'localhost', port: 3000),
  });

  DroneState copyWith({
    ConnectionStatus? connectionStatus,
    int? batteryLevel,
    String? lastCommandStatus,
    String? lastCommandMessage,
    bool? isExecutingCommand,
    ServerConfig? serverConfig,
  }) {
    return DroneState(
      connectionStatus: connectionStatus ?? this.connectionStatus,
      batteryLevel: batteryLevel ?? this.batteryLevel,
      lastCommandStatus: lastCommandStatus ?? this.lastCommandStatus,
      lastCommandMessage: lastCommandMessage ?? this.lastCommandMessage,
      isExecutingCommand: isExecutingCommand ?? this.isExecutingCommand,
      serverConfig: serverConfig ?? this.serverConfig,
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
  ];
}