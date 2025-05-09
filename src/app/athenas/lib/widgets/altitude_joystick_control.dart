import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';
import '../models/drone_command.dart';
import 'drone_joystick_control.dart';

class AltitudeJoystickControl extends StatefulWidget {
  final Color baseColor;
  final Color stickColor;
  final double size;

  const AltitudeJoystickControl({
    Key? key,
    this.baseColor = const Color(0x99FFFFFF),
    this.stickColor = Colors.white,
    this.size = 150,
  }) : super(key: key);

  @override
  State<AltitudeJoystickControl> createState() => _AltitudeJoystickControlState();
}

class _AltitudeJoystickControlState extends State<AltitudeJoystickControl> {
  Timer? _movementTimer;
  JoystickDirectionEnum _lastDirection = JoystickDirectionEnum.idle;
  final int _altitudeDistance = 30; // Default altitude change in cm

  @override
  void dispose() {
    _movementTimer?.cancel();
    super.dispose();
  }

  // Convertendo valores Y do joystick para direção vertical apenas
  JoystickDirectionEnum _getVerticalDirectionFromOffset(double x, double y) {
    // Definir um limiar para detectar movimento
    const double threshold = 0.3;
    
    if (y.abs() < threshold) {
      return JoystickDirectionEnum.idle;
    }
    
    // Apenas nos interessa o movimento vertical para altitude
    return y < 0 ? JoystickDirectionEnum.up : JoystickDirectionEnum.down;
  }

  void _processJoystickInput(JoystickDirectionEnum direction) {
    // Apenas considerar movimentos verticais para controle de altitude
    JoystickDirectionEnum effectiveDirection;
    if (direction == JoystickDirectionEnum.up) {
      effectiveDirection = JoystickDirectionEnum.up;
    } else if (direction == JoystickDirectionEnum.down) {
      effectiveDirection = JoystickDirectionEnum.down;
    } else {
      effectiveDirection = JoystickDirectionEnum.idle;
    }
    
    // Prevent unnecessary updates if direction hasn't changed
    if (effectiveDirection == _lastDirection) {
      return;
    }

    // Cancel any existing timer
    _movementTimer?.cancel();

    // If joystick is idle, don't start a new timer
    if (effectiveDirection == JoystickDirectionEnum.idle) {
      _lastDirection = effectiveDirection;
      return;
    }

    // Update the last direction
    _lastDirection = effectiveDirection;
    
    // Send the command immediately
    _sendDroneCommand(effectiveDirection);

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

    // Special commands for altitude control
    DroneCommand command;
    
    switch (direction) {
      case JoystickDirectionEnum.up:
        command = MoveCommand('up', _altitudeDistance);
        break;
      case JoystickDirectionEnum.down:
        command = MoveCommand('down', _altitudeDistance);
        break;
      default:
        return; // No command for other directions
    }

    context.read<DroneBloc>().add(SendCommandEvent(command));
  }

  @override
  Widget build(BuildContext context) {
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
          const Text(
            'Altitude',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
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
                // O StickDragDetails fornece valores x e y diretamente
                final direction = _getVerticalDirectionFromOffset(details.x, details.y);
                _processJoystickInput(direction);
              },
            ),
          ),
        ],
      ),
    );
  }
}