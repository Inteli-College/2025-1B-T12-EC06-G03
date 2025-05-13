import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';
import 'drone_joystick_control.dart';

class AltitudeJoystickControl extends StatefulWidget {
  final double size;
  final GlobalKey<DroneJoystickControlState>? mainJoystickKey;

  const AltitudeJoystickControl({
    Key? key,
    required this.size,
    this.mainJoystickKey,
  }) : super(key: key);

  @override
  State<AltitudeJoystickControl> createState() =>
      _AltitudeJoystickControlState();
}

class _AltitudeJoystickControlState extends State<AltitudeJoystickControl> {
  // Valores do joystick
  double _ud = 0; // up/down
  double _yw = 0; // yaw rotation

  Timer? _rcControlTimer;
  bool _isJoystickActive = false;

  @override
  void initState() {
    super.initState();

    // Iniciar um timer que atualiza os comandos a uma taxa fixa
    _rcControlTimer = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      if (_isJoystickActive) {
        // Atualizar o joystick principal com os valores de altitude e rotação
        _updateMainJoystick();

        // Se o joystick principal não tiver uma chave associada ou estiver inativo,
        // enviamos comandos diretamente daqui
        if (widget.mainJoystickKey == null ||
            !widget.mainJoystickKey!.currentState!.isJoystickActive) {
          _sendRCControl();
        }
      }
    });
  }

  @override
  void dispose() {
    _rcControlTimer?.cancel();
    super.dispose();
  }

  // Converte os valores do joystick (-1 a 1) para a escala correta (-100 a 100)
  // com uma curva de resposta quadrática para melhor controle em velocidades mais baixas
  int _scaleJoystickValue(double value) {
    // Aplicar uma curva quadrática para melhorar a precisão em baixas velocidades
    // Preservando o sinal do valor original
    double sign = value >= 0 ? 1 : -1;
    double absValue = value.abs();
    
    // Curva quadrática suave: valor^2 para melhor controle em baixas velocidades
    double scaled = sign * (absValue * absValue) * 100;
    
    // Arredondar para o inteiro mais próximo
    return scaled.round();
  }

  // Atualiza o joystick principal com os valores de altitude e rotação
  void _updateMainJoystick() {
    if (widget.mainJoystickKey?.currentState != null) {
      widget.mainJoystickKey!.currentState!.updateAltitudeAndYaw(_ud, _yw);
    }
  }

  // Envia os valores de controle ao drone
  void _sendRCControl() {
    // Se o joystick não está ativo, não enviamos comando
    if (!_isJoystickActive) return;

    // Enviamos apenas os valores controlados por este joystick
    // Outros valores (lr, fb) são mantidos em zero
    final params = {
      'lr': 0,
      'fb': 0,
      'ud': _scaleJoystickValue(_ud),
      'yw': _scaleJoystickValue(_yw),
    };

    context.read<DroneBloc>().add(SendRCControlEvent(params));
  }

  // Para o drone enviando zeros para os controles deste joystick
  void _stopAltitudeAndYaw() {
    final params = {
      'lr': 0,
      'fb': 0,
      'ud': 0,
      'yw': 0,
    };

    context.read<DroneBloc>().add(SendRCControlEvent(params));

    // Resetar os valores internos
    _ud = 0;
    _yw = 0;

    // Atualizar o joystick principal
    _updateMainJoystick();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // Joystick background with visual indicators
        Container(
          width: widget.size,
          height: widget.size,
          decoration: BoxDecoration(
            color: Colors.black54,
            shape: BoxShape.circle,
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Vertical arrows (altitude)
              Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: EdgeInsets.only(top: widget.size * 0.15),
                    child: Icon(
                      Icons.arrow_upward,
                      color: Colors.white38,
                      size: widget.size * 0.15,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(bottom: widget.size * 0.15),
                    child: Icon(
                      Icons.arrow_downward,
                      color: Colors.white38,
                      size: widget.size * 0.15,
                    ),
                  ),
                ],
              ),
              // Horizontal arrows (yaw/rotation)
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: EdgeInsets.only(left: widget.size * 0.15),
                    child: Icon(
                      Icons.rotate_left,
                      color: Colors.white38,
                      size: widget.size * 0.15,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(right: widget.size * 0.15),
                    child: Icon(
                      Icons.rotate_right,
                      color: Colors.white38,
                      size: widget.size * 0.15,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),

        // Actual joystick control
        Container(
          width: widget.size,
          height: widget.size,
          child: JoystickArea(
            listener: (details) {
              setState(() {
                // Define deadzone to prevent small unintentional movements (5% of range)
                const double DEADZONE = 0.05;
                
                // Verificar se valores são válidos antes de usar e aplicar deadzone
                double rawX = !details.x.isNaN ? details.x : 0.0;
                double rawY = !details.y.isNaN ? details.y : 0.0;
                
                // Apply deadzone to X (rotation/yaw)
                double validX = rawX.abs() < DEADZONE ? 0.0 : rawX;
                
                // Apply deadzone to Y (up/down)
                double validY = rawY.abs() < DEADZONE ? 0.0 : rawY;
                
                // Mapear Y para ud (up-down) com validação
                _ud = -validY; // Invertido para que para cima seja positivo

                // Mapear X para yw (yaw - rotação) com validação
                _yw = validX;

                // Ativar o joystick quando em uso (com valores válidos)
                _isJoystickActive = (validX != 0 || validY != 0);

                // Se o joystick foi "solto" (voltou ao centro)
                if (!_isJoystickActive) {
                  _stopAltitudeAndYaw();
                }
              });
            },
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.2),
                shape: BoxShape.circle,
                border: Border.all(
                  color: _isJoystickActive 
                      ? Colors.orangeAccent.withOpacity(0.8) 
                      : Colors.white24,
                  width: 2.0,
                ),
                boxShadow: _isJoystickActive
                    ? [
                        BoxShadow(
                          color: Colors.orangeAccent.withOpacity(0.5),
                          blurRadius: 10,
                          spreadRadius: 2,
                        )
                      ]
                    : [],
              ),
              child: Center(
                child: Container(
                  width: widget.size * 0.3,
                  height: widget.size * 0.3,
                  decoration: BoxDecoration(
                    color: _isJoystickActive 
                        ? Colors.orange.withOpacity(0.7) 
                        : Colors.black.withOpacity(0.5),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black38,
                        blurRadius: 5,
                        spreadRadius: 1,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Icon(
                    Icons.height,
                    color: Colors.white,
                    size: widget.size * 0.15,
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
