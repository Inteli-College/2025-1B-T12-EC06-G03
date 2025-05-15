import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';

class DroneJoystickControl extends StatefulWidget {
  final double size;

  const DroneJoystickControl({
    Key? key,
    required this.size,
  }) : super(key: key);

  @override
  DroneJoystickControlState createState() => DroneJoystickControlState();
}

// State class made public so it can be referenced from other files
class DroneJoystickControlState extends State<DroneJoystickControl> {
  // Valores do joystick
  double _lr = 0; // left/right
  double _fb = 0; // forward/backward
  double _ud = 0; // up/down (controlado pelo altitude_joystick_control.dart)
  double _yw = 0; // yaw rotation

  Timer? _rcControlTimer;
  bool _isJoystickActive = false;

  // Método público para verificar se o joystick está ativo
  bool get isJoystickActive => _isJoystickActive;

  @override
  void initState() {
    super.initState();

    // Iniciar um timer que envia os comandos RC a cada 50ms
    // Uma taxa de atualização mais consistente para melhor resposta
    _rcControlTimer = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      if (_isJoystickActive) {
        _sendRCControl();
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

  // Envia os valores de controle ao drone
  void _sendRCControl() {
    // Se o joystick não está ativo, não enviamos comando
    if (!_isJoystickActive) return;

    // Valores lr, fb são definidos pelo joystick principal
    // Valor ud (up/down) e yw (yaw) serão recebidos do outro joystick
    final params = {
      'lr': _scaleJoystickValue(_lr),
      'fb': _scaleJoystickValue(
          -_fb), // Invertido para corresponder à convenção do backend
      'ud': _scaleJoystickValue(_ud),
      'yw': _scaleJoystickValue(_yw),
    };

    context.read<DroneBloc>().add(SendRCControlEvent(params));
  }

  // Para o drone enviando zeros para todos os controles
  void _stopDrone() {
    final params = {
      'lr': 0,
      'fb': 0,
      'ud': 0,
      'yw': 0,
    };

    context.read<DroneBloc>().add(SendRCControlEvent(params));

    // Resetar os valores internos
    _lr = 0;
    _fb = 0;
    // Não resetamos ud e yw pois são controlados pelo outro joystick
  }

  // Método público para receber os valores de altitude e rotação
  void updateAltitudeAndYaw(double ud, double yw) {
    setState(() {
      // Validar os valores antes de usar
      _ud = ud.isNaN ? 0.0 : ud;
      _yw = yw.isNaN ? 0.0 : yw;
    });
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
              // Vertical arrows (forward/backward)
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
              // Horizontal arrows (left/right)
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: EdgeInsets.only(left: widget.size * 0.15),
                    child: Icon(
                      Icons.arrow_back,
                      color: Colors.white38,
                      size: widget.size * 0.15,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(right: widget.size * 0.15),
                    child: Icon(
                      Icons.arrow_forward,
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
                
                // Apply deadzone to X (left-right)
                double rawX = details.x.isNaN ? 0 : details.x;
                _lr = rawX.abs() < DEADZONE ? 0 : rawX;
                
                // Apply deadzone to Y (forward-backward)
                double rawY = details.y.isNaN ? 0 : details.y;
                _fb = rawY.abs() < DEADZONE ? 0 : rawY;

                // Ativar o joystick quando em uso (verifica se valores são válidos)
                _isJoystickActive = (_lr != 0 || _fb != 0);

                // Se o joystick foi "solto" (voltou ao centro)
                if (!_isJoystickActive) {
                  _stopDrone();
                }
              });
            },
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.2),
                shape: BoxShape.circle,
                border: Border.all(
                  color: _isJoystickActive 
                      ? Colors.lightBlueAccent.withOpacity(0.8) 
                      : Colors.white24,
                  width: 2.0,
                ),
                boxShadow: _isJoystickActive
                    ? [
                        BoxShadow(
                          color: Colors.lightBlueAccent.withOpacity(0.5),
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
                        ? Colors.blue.withOpacity(0.7) 
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
                    Icons.flight,
                    color: Colors.white,
                    size: widget.size * 0.15,
                  ),
                ),
              ),
            ),
          ),
        )
      ],
    );
  }
}
