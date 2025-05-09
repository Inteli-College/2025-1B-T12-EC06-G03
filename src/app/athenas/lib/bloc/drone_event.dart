import 'package:equatable/equatable.dart';
import '../models/drone_command.dart';
import '../models/drone_response.dart';
import '../models/server_config.dart';

abstract class DroneEvent extends Equatable {
  const DroneEvent();

  @override
  List<Object?> get props => [];
}

class ConnectEvent extends DroneEvent {}

class DisconnectEvent extends DroneEvent {}

class UpdateServerConfigEvent extends DroneEvent {
  final ServerConfig serverConfig;

  const UpdateServerConfigEvent(this.serverConfig);

  @override
  List<Object?> get props => [serverConfig];
}

class SendCommandEvent extends DroneEvent {
  final DroneCommand command;

  const SendCommandEvent(this.command);

  @override
  List<Object?> get props => [command];
}

class ResponseReceivedEvent extends DroneEvent {
  final DroneResponse response;

  const ResponseReceivedEvent(this.response);

  @override
  List<Object?> get props => [response];
}

class RequestBatteryLevelEvent extends DroneEvent {}