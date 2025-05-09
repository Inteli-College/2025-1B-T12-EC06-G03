import 'dart:async';
import 'package:bloc/bloc.dart';
import '../models/drone_command.dart';
import '../models/drone_response.dart';
import '../services/drone_service.dart';
import 'drone_event.dart';
import 'drone_state.dart';

class DroneBloc extends Bloc<DroneEvent, DroneState> {
  final DroneService _droneService;
  late StreamSubscription<DroneResponse> _droneResponseSubscription;

  DroneBloc(this._droneService) : super(const DroneState()) {
    on<ConnectEvent>(_onConnect);
    on<DisconnectEvent>(_onDisconnect);
    on<SendCommandEvent>(_onSendCommand);
    on<ResponseReceivedEvent>(_onResponseReceived);
    on<RequestBatteryLevelEvent>(_onRequestBatteryLevel);
    on<UpdateServerConfigEvent>(_onUpdateServerConfig);

    // Listen to responses from the drone service
    _droneResponseSubscription = _droneService.responseStream.listen(
      (response) => add(ResponseReceivedEvent(response)),
    );
  }

  FutureOr<void> _onConnect(ConnectEvent event, Emitter<DroneState> emit) {
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.connecting,
    ));
  }

  FutureOr<void> _onDisconnect(DisconnectEvent event, Emitter<DroneState> emit) {
    _droneService.disconnect();
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.disconnected,
    ));
  }

  FutureOr<void> _onUpdateServerConfig(UpdateServerConfigEvent event, Emitter<DroneState> emit) {
    final newConfig = event.serverConfig;
    
    // Update state first
    emit(state.copyWith(
      serverConfig: newConfig,
      connectionStatus: ConnectionStatus.disconnected,
    ));
    
    // Update service configuration
    _droneService.updateServerConfig(
      host: newConfig.host,
      port: newConfig.port,
    );
    
    // Change state to connecting since updateServerConfig initiates a new connection
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.connecting,
    ));
  }

  FutureOr<void> _onSendCommand(SendCommandEvent event, Emitter<DroneState> emit) async {
    emit(state.copyWith(
      isExecutingCommand: true,
      lastCommandStatus: null,
      lastCommandMessage: null,
    ));
    
    await _droneService.sendCommand(event.command);
  }

  FutureOr<void> _onResponseReceived(ResponseReceivedEvent event, Emitter<DroneState> emit) {
    final response = event.response;
    
    if (response.command == 'connect') {
      emit(state.copyWith(
        connectionStatus: ConnectionStatus.connected,
        lastCommandStatus: response.status,
        lastCommandMessage: response.message,
        isExecutingCommand: false,
      ));
    } else if (response.command == 'disconnect') {
      emit(state.copyWith(
        connectionStatus: ConnectionStatus.disconnected,
        lastCommandStatus: response.status,
        lastCommandMessage: response.message,
        isExecutingCommand: false,
      ));
    } else if (response.command == 'battery') {
      // Battery response typically contains a numeric value
      final batteryLevel = response.response is int 
          ? response.response as int
          : response.response is String
              ? int.tryParse(response.response as String)
              : null;
              
      emit(state.copyWith(
        batteryLevel: batteryLevel,
        lastCommandStatus: response.status,
        lastCommandMessage: response.message,
        isExecutingCommand: false,
      ));
    } else {
      // For other commands, update the command status and message
      emit(state.copyWith(
        lastCommandStatus: response.status,
        lastCommandMessage: response.message,
        isExecutingCommand: false,
      ));
    }
  }

  FutureOr<void> _onRequestBatteryLevel(RequestBatteryLevelEvent event, Emitter<DroneState> emit) {
    emit(state.copyWith(isExecutingCommand: true));
    _droneService.sendCommand(const BatteryCommand());
  }
  
  String getVideoStreamUrl() {
    return _droneService.getVideoStreamUrl();
  }

  @override
  Future<void> close() {
    _droneResponseSubscription.cancel();
    _droneService.dispose();
    return super.close();
  }
}