import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_state.dart';

class StatusNotification extends StatefulWidget {
  const StatusNotification({Key? key}) : super(key: key);

  @override
  State<StatusNotification> createState() => _StatusNotificationState();
}

class _StatusNotificationState extends State<StatusNotification> {
  bool _showNotification = true;
  Timer? _timer;
  String? _lastStatusKey;
  
  @override
  void initState() {
    super.initState();
    // Inicializa com a notificação visível
    _showNotification = true;
  }
  
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Reinicia estado se dependências mudarem (por exemplo, ao trocar de tela)
    _resetNotificationState();
  }
  
  @override
  void dispose() {
    // Cancela o timer para evitar memory leaks
    _timer?.cancel();
    _timer = null;
    super.dispose();
  }
  
  void _resetNotificationState() {
    _timer?.cancel();
    _timer = null;
    _lastStatusKey = null;
    _showNotification = true;
  }
  
  void _startTimer(String statusKey) {
    // Sempre cancela o timer existente para evitar múltiplas chamadas
    _timer?.cancel();
    _timer = null;
    
    // Verifica se a mensagem é nova e atualiza a visibilidade
    if (_lastStatusKey != statusKey) {
      _lastStatusKey = statusKey;
      // Atualiza o estado de visibilidade sem chamar setState durante o build
      _showNotification = true;
    }
    
    // Configura um novo timer para esconder a notificação após 5 segundos
    _timer = Timer(const Duration(seconds: 5), () {
      // Verifica se o widget ainda está montado antes de atualizar o estado
      if (mounted) {
        // Usa setState de forma segura após o build estar completo
        setState(() {
          _showNotification = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final isSmallScreen = screenSize.width < 600;
    
    // Ajustes responsivos
    final fontSize = isSmallScreen ? 12.0 : 14.0;
    final iconSize = isSmallScreen ? 16.0 : 20.0;
    final paddingHorizontal = isSmallScreen ? 12.0 : 24.0;
    final paddingVertical = isSmallScreen ? 6.0 : 10.0;
    
    return BlocBuilder<DroneBloc, DroneState>(
      buildWhen: (previous, current) => 
          previous.lastCommandStatus != current.lastCommandStatus ||
          previous.lastCommandMessage != current.lastCommandMessage ||
          previous.isExecutingCommand != current.isExecutingCommand,
      builder: (context, state) {
        // Não mostra nada se não houver status de comando
        if (state.lastCommandStatus == null) {
          return const SizedBox.shrink();
        }

        // Verifica se é um comando bem-sucedido
        // Usa null check seguro para evitar NPE
        final isSuccess = state.lastCommandStatus == 'success' ||
            state.lastCommandStatus == 'ok' ||
            (state.lastCommandMessage != null && 
             state.lastCommandMessage!.contains('Connected'));
             
        final isExecuting = state.isExecutingCommand;
        
        // Não mostra notificação para comandos bem-sucedidos (exceto comandos em execução)
        if (isSuccess && !isExecuting) {
          return const SizedBox.shrink();
        }
        
        // Gera uma chave única para esta notificação
        // Usa uma combinação estável de status e mensagem
        final statusKey = "${state.lastCommandStatus}-${state.lastCommand ?? 'no-command'}-${state.lastCommandMessage ?? 'no-message'}";
        
        // Usando um post-frame callback para evitar chamar setState durante o build
        // Usando um bool para evitar chamadas repetidas ao mesmo status
        if (_lastStatusKey != statusKey) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            _startTimer(statusKey);
          });
        }
        
        // Se a notificação deve estar oculta, retorna widget vazio
        if (!_showNotification) {
          return const SizedBox.shrink();
        }
        
        Color bgColor;
        Color textColor;
        IconData iconData;
        String statusText;

        if (isExecuting) {
          bgColor = Colors.blue.shade600;
          textColor = Colors.white;
          iconData = Icons.pending_outlined;
          statusText = "Executando...";
        } else if (isSuccess) {
          bgColor = Colors.green.shade600;
          textColor = Colors.white;
          iconData = Icons.check_circle_outline;
          statusText = "Executado com sucesso";
        } else {
          bgColor = Colors.red.shade600;
          textColor = Colors.white;
          iconData = Icons.error_outline;
          statusText = "Erro ao executar comando";
        }

        return Center(
          child: AnimatedOpacity(
            opacity: _showNotification ? 1.0 : 0.0,
            duration: const Duration(milliseconds: 500),
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              transitionBuilder: (Widget child, Animation<double> animation) {
                return FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween<Offset>(
                      begin: const Offset(0, 1),
                      end: Offset.zero,
                    ).animate(animation),
                    child: child,
                  ),
                );
              },
              child: Container(
                key: ValueKey<String>(statusKey),
                margin: EdgeInsets.symmetric(horizontal: screenSize.width * 0.1),
                padding: EdgeInsets.symmetric(
                  horizontal: paddingHorizontal,
                  vertical: paddingVertical,
                ),
                decoration: BoxDecoration(
                  color: bgColor,
                  borderRadius: BorderRadius.circular(30),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 8,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      iconData,
                      color: textColor,
                      size: iconSize,
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        "$statusText: ${state.lastCommandMessage ?? 'Sem detalhes'}",
                        style: TextStyle(
                          color: textColor,
                          fontWeight: FontWeight.bold,
                          fontSize: fontSize,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}