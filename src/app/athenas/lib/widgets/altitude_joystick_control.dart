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
  final int _rotationDegree = 40; // Default rotation in degrees

  @override
  void dispose() {
    _movementTimer?.cancel();
    super.dispose();
  }

  // Convertendo valores do joystick para direção vertical e horizontal
  JoystickDirectionEnum _getDirectionFromOffset(double x, double y) {
    // Definir um limiar para detectar movimento
    const double threshold = 0.3;
    
    if (x.abs() < threshold && y.abs() < threshold) {
      return JoystickDirectionEnum.idle;
    }
    
    // Se o movimento horizontal for mais significativo que o vertical
    if (x.abs() > y.abs()) {
      return x > 0 ? JoystickDirectionEnum.right : JoystickDirectionEnum.left;
    } 
    // Se o movimento vertical for mais significativo
    else {
      return y < 0 ? JoystickDirectionEnum.up : JoystickDirectionEnum.down;
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
    
    // Send the command immediately
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

    // Commands for altitude control and rotation
    DroneCommand command;
    
    switch (direction) {
      case JoystickDirectionEnum.up:
        command = MoveCommand('up', _altitudeDistance);
        break;
      case JoystickDirectionEnum.down:
        command = MoveCommand('down', _altitudeDistance);
        break;
      case JoystickDirectionEnum.left:
        command = RotateCommand('ccw', _rotationDegree);
        break;
      case JoystickDirectionEnum.right:
        command = RotateCommand('cw', _rotationDegree);
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
            'Altitude & Rotation',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 5),
          Expanded(
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Indicator arrows for direction
                Positioned(
                  top: 0,
                  child: Icon(Icons.arrow_upward, color: Colors.white54, size: 14),
                ),
                Positioned(
                  bottom: 0,
                  child: Icon(Icons.arrow_downward, color: Colors.white54, size: 14),
                ),
                Positioned(
                  left: 0,
                  child: Row(
                    children: [
                      Icon(Icons.rotate_left, color: Colors.white54, size: 14),
                      SizedBox(width: 2),
                      Text('CCW', style: TextStyle(color: Colors.white54, fontSize: 10)),
                    ],
                  ),
                ),
                Positioned(
                  right: 0,
                  child: Row(
                    children: [
                      Text('CW', style: TextStyle(color: Colors.white54, fontSize: 10)),
                      SizedBox(width: 2),
                      Icon(Icons.rotate_right, color: Colors.white54, size: 14),
                    ],
                  ),
                ),
                // The actual joystick
                Joystick(
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
              ],
            ),
          ),
        ],
      ),
    );
  }
}