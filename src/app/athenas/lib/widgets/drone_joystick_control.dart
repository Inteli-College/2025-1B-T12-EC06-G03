import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';
import '../models/drone_command.dart';

// Definição de direção do joystick
enum JoystickDirectionEnum { up, down, left, right, upLeft, upRight, downLeft, downRight, idle }

class DroneJoystickControl extends StatefulWidget {
  final Color baseColor;
  final Color stickColor;
  final double size;
  final String label;

  const DroneJoystickControl({
    Key? key,
    this.baseColor = const Color(0x99FFFFFF),
    this.stickColor = Colors.white,
    this.size = 150,
    required this.label,
  }) : super(key: key);

  @override
  State<DroneJoystickControl> createState() => _DroneJoystickControlState();
}

class _DroneJoystickControlState extends State<DroneJoystickControl> {
  Timer? _movementTimer;
  JoystickDirectionEnum _lastDirection = JoystickDirectionEnum.idle;
  int _movementDistance = 30; // Default distance in cm for each movement
  
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Ajusta a distância de movimento com base no tamanho da tela
    final screenSize = MediaQuery.of(context).size;
    final smallestDimension = screenSize.shortestSide;
    
    // Ajusta a distância conforme o tamanho da tela
    if (smallestDimension < 400) {
      _movementDistance = 20; // Para telas menores, movimentos mais curtos
    } else if (smallestDimension > 800) {
      _movementDistance = 40; // Para telas maiores, movimentos mais longos
    }
  }

  @override
  void dispose() {
    _movementTimer?.cancel();
    super.dispose();
  }

  // Convertendo valores X e Y do joystick para direção
  JoystickDirectionEnum _getDirectionFromOffset(double x, double y) {
    // Definir um limiar para detectar movimento
    const double threshold = 0.5;
    const double diagonalThreshold = 0.3;
    
    if (x.abs() < threshold && y.abs() < threshold) {
      return JoystickDirectionEnum.idle;
    }
    
    // Detectar movimentos diagonais quando ambos x e y são significativos
    if (x.abs() > diagonalThreshold && y.abs() > diagonalThreshold) {
      // Movimentos diagonais
      if (x > 0 && y < 0) return JoystickDirectionEnum.upRight;
      if (x < 0 && y < 0) return JoystickDirectionEnum.upLeft;
      if (x > 0 && y > 0) return JoystickDirectionEnum.downRight;
      if (x < 0 && y > 0) return JoystickDirectionEnum.downLeft;
    }
    
    // Caso não seja diagonal, continua com a lógica anterior
    if (x.abs() > y.abs()) {
      // Movimento horizontal predominante
      return x > 0 
          ? JoystickDirectionEnum.right 
          : JoystickDirectionEnum.left;
    } else {
      // Movimento vertical predominante
      return y > 0 
          ? JoystickDirectionEnum.down 
          : JoystickDirectionEnum.up;
    }
  }

  void _processJoystickInput(JoystickDirectionEnum direction) {
    // Prevent unnecessary updates if direction hasn't changed
    if (direction == _lastDirection) {
      return;
    }

    // Cancel any existing timer
    _movementTimer?.cancel();

    // If joystick is idle, don't start a new timer
    if (direction == JoystickDirectionEnum.idle) {
      _lastDirection = direction;
      return;
    }

    // Update the last direction
    _lastDirection = direction;
    
    // Convert joystick direction to drone command
    _sendDroneCommand(direction);

    // Start a timer to continuously send commands while joystick is held
    _movementTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (_lastDirection != JoystickDirectionEnum.idle) {
        _sendDroneCommand(_lastDirection);
      } else {
        timer.cancel();
      }
    });
  }

  void _sendDroneCommand(JoystickDirectionEnum direction) {
    // Skip if in idle position
    if (direction == JoystickDirectionEnum.idle) {
      return;
    }

    // Map joystick direction to drone command
    DroneCommand command;
    
    switch (direction) {
      case JoystickDirectionEnum.up:
        command = MoveCommand('forward', _movementDistance);
        break;
      case JoystickDirectionEnum.right:
        command = MoveCommand('right', _movementDistance);
        break;
      case JoystickDirectionEnum.down:
        command = MoveCommand('back', _movementDistance);
        break;
      case JoystickDirectionEnum.left:
        command = MoveCommand('left', _movementDistance);
        break;
      case JoystickDirectionEnum.upRight:
        command = RotateCommand('cw', 45);
        break;
      case JoystickDirectionEnum.downRight:
        command = RotateCommand('cw', 45);
        break;
      case JoystickDirectionEnum.downLeft:
        command = RotateCommand('ccw', 45);
        break;
      case JoystickDirectionEnum.upLeft:
        command = RotateCommand('ccw', 45);
        break;
      case JoystickDirectionEnum.idle:
        return; // No command for idle
    }

    context.read<DroneBloc>().add(SendCommandEvent(command));
  }

  @override
  Widget build(BuildContext context) {
    // Ajustar tamanho da fonte baseado no tamanho do joystick
    final fontSize = widget.size < 120 ? 10.0 : 12.0;
    
    return Container(
      width: widget.size,
      height: widget.size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: Colors.black45,
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            widget.label,
            style: TextStyle(
              color: Colors.white,
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 5),
          Expanded(
            child: Joystick(
              base: JoystickBase(
                size: widget.size * 0.8,
              ),
              stick: JoystickStick(
                size: widget.size * 0.3,
              ),
              listener: (details) {
                final direction = _getDirectionFromOffset(details.x, details.y);
                _processJoystickInput(direction);
              },
            ),
          ),
        ],
      ),
    );
  }
}