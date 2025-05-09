import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_state.dart';

class StatusNotification extends StatelessWidget {
  const StatusNotification({Key? key}) : super(key: key);

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
          previous.lastCommandMessage != current.lastCommandMessage,
      builder: (context, state) {
        if (state.lastCommandStatus == null) {
          return const SizedBox.shrink();
        }

        final isSuccess = state.lastCommandStatus == 'success' ||
            state.lastCommandStatus == 'ok' ||
            state.lastCommandMessage!.contains('Connected');
        final isExecuting = state.isExecutingCommand;
        
        // Don't show notification for successful commands
        if (isSuccess && !isExecuting) {
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
              key: ValueKey<String>("${state.lastCommandStatus}"),
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
        );
      },
    );
  }
}