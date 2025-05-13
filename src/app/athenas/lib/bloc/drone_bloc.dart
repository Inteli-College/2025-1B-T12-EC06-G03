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
  Timer? _batteryCheckTimer;

  DroneBloc(this._droneService) : super(DroneState.initial()) {
    _droneResponseSubscription = _droneService.responseStream.listen((response) {
      add(ResponseReceivedEvent(response));
    });

    on<ConnectEvent>(_onConnectEvent);
    on<DisconnectEvent>(_onDisconnectEvent);
    on<UpdateServerConfigEvent>(_onUpdateServerConfigEvent);
    on<SendCommandEvent>(_onSendCommandEvent);
    on<ResponseReceivedEvent>(_onResponseReceivedEvent);
    on<RequestBatteryLevelEvent>(_onRequestBatteryLevelEvent);
    on<SendRCControlEvent>(_onSendRCControlEvent); // Novo handler para controle RC
  }

  Future<void> _onConnectEvent(ConnectEvent event, Emitter<DroneState> emit) async {
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.connecting,
    ));
    
    // A conexão real acontece via Socket.IO quando o servidor é configurado
    // Iniciamos o temporizador para verificação periódica da bateria
    _startBatteryCheckTimer();
  }

  Future<void> _onDisconnectEvent(DisconnectEvent event, Emitter<DroneState> emit) async {
    _droneService.disconnect();
    _batteryCheckTimer?.cancel();
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.disconnected,
    ));
  }

  Future<void> _onUpdateServerConfigEvent(UpdateServerConfigEvent event, Emitter<DroneState> emit) async {
    emit(state.copyWith(
      connectionStatus: ConnectionStatus.connecting,
      serverConfig: event.serverConfig,
    ));
    
    _droneService.updateServerConfig(
      host: event.serverConfig.host,
      port: event.serverConfig.port,
    );
  }

  Future<void> _onSendCommandEvent(SendCommandEvent event, Emitter<DroneState> emit) async {
    // Atualiza o estado para indicar que um comando está sendo executado
    emit(state.copyWith(
      isExecutingCommand: true,
      lastCommand: event.command.command
    ));
    
    await _droneService.sendCommand(event.command);
  }

  Future<void> _onResponseReceivedEvent(ResponseReceivedEvent event, Emitter<DroneState> emit) async {
    final response = event.response;
    
    switch (response.command) {
      case 'connect':
        emit(state.copyWith(
          connectionStatus: response.status == 'ok' ? ConnectionStatus.connected : ConnectionStatus.disconnected,
          lastCommandStatus: response.status,
          lastCommandMessage: response.message,
        ));
        break;
      
      case 'battery':
        // Melhorar o tratamento da resposta do nível de bateria
        int batteryLevel = 0;
        
        try {
          // Primeiro verificar se o response tem um campo 'response' com o valor
          if (response.response != null) {
            // Tentar extrair o nível da bateria do campo response
            if (response.response is int) {
              batteryLevel = response.response;
            } else if (response.response is String && int.tryParse(response.response) != null) {
              batteryLevel = int.parse(response.response);
            }
          }
          
          // Se não encontrou no campo response, tentar no campo message
          if (batteryLevel == 0 && response.message != null) {
            String responseMsg = response.message ?? '';
            if (responseMsg.isNotEmpty) {
              // Se for apenas números, converter diretamente
              if (int.tryParse(responseMsg) != null) {
                batteryLevel = int.parse(responseMsg);
              } else {
                // Tentar extrair apenas os dígitos da resposta
                final RegExp regex = RegExp(r'\d+');
                final match = regex.firstMatch(responseMsg);
                if (match != null) {
                  batteryLevel = int.tryParse(match.group(0) ?? '0') ?? 0;
                }
              }
            }
          }
          
          print('Received battery level: $batteryLevel from response: ${response.response}, message: ${response.message}');
        } catch (e) {
          print('Error processing battery level: $e');
        }
        
        emit(state.copyWith(
          batteryLevel: batteryLevel,
          lastCommandStatus: response.status,
          lastCommandMessage: 'Nível de bateria: $batteryLevel%',
          isExecutingCommand: false,
        ));
        break;

      case 'rc_control':
        // Resposta ao controle RC
        emit(state.copyWith(
          lastCommandStatus: response.status,
          lastCommandMessage: response.message,
          isExecutingCommand: false,
        ));
        break;
      
      default:
        // Para todos os outros comandos, apenas atualizamos o status
        emit(state.copyWith(
          lastCommandStatus: response.status,
          lastCommandMessage: response.message,
          isExecutingCommand: false,
        ));
        break;
    }
  }

  Future<void> _onRequestBatteryLevelEvent(RequestBatteryLevelEvent event, Emitter<DroneState> emit) async {
    // Verificar se já está conectado ao drone
    if (state.connectionStatus != ConnectionStatus.connected) {
      print('Não é possível solicitar bateria: drone desconectado');
      return;
    }
    
    print('Solicitando nível de bateria do drone');
    
    // Atualiza o estado para indicar que está solicitando bateria
    emit(state.copyWith(
      isExecutingCommand: true,
      lastCommand: 'battery'
    ));
    
    try {
      // Envia o comando de bateria
      await _droneService.sendCommand(const BatteryCommand());
      print('Comando de bateria enviado com sucesso');
    } catch (e) {
      print('Erro ao enviar comando de bateria: $e');
      
      // Atualiza estado em caso de erro
      emit(state.copyWith(
        isExecutingCommand: false,
        lastCommandStatus: 'error',
        lastCommandMessage: 'Erro ao solicitar bateria: $e'
      ));
    }
  }

  // Handler para o evento de controle RC com tratamento eficiente
  Future<void> _onSendRCControlEvent(SendRCControlEvent event, Emitter<DroneState> emit) async {
    if (state.connectionStatus != ConnectionStatus.connected) return;
    
    // Verificar se os valores do RC são todos zero (parado)
    bool isDroneStopped = event.params.values.every((value) => value == 0);
    
    // Apenas atualizamos o estado se o drone está se movendo
    // Isso reduz as atualizações desnecessárias do estado
    if (!isDroneStopped) {
      emit(state.copyWith(
        lastCommand: 'rc_control'
      ));
    }
    
    // Enviar comando para o serviço
    await _droneService.sendRCControl(event.params);
  }

  void _startBatteryCheckTimer() {
    _batteryCheckTimer?.cancel();
    
    // Solicitar imediatamente o nível da bateria ao conectar
    add(RequestBatteryLevelEvent());
    
    // Configurar verificação periódica a cada 15 segundos (mais frequente)
    _batteryCheckTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => add(RequestBatteryLevelEvent()),
    );
  }

  String getVideoStreamUrl() => _droneService.getVideoStreamUrl();
  
  bool get isConnected => state.connectionStatus == ConnectionStatus.connected;

  @override
  Future<void> close() {
    _droneResponseSubscription.cancel();
    _batteryCheckTimer?.cancel();
    _droneService.dispose();
    return super.close();
  }
}