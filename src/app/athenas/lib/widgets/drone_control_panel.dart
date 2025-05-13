import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:material_design_icons_flutter/material_design_icons_flutter.dart';

import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';
import '../bloc/drone_state.dart';
import '../models/drone_command.dart';

class DroneControlPanel extends StatelessWidget {
  const DroneControlPanel({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<DroneBloc, DroneState>(
      builder: (context, state) {
        final isConnected = state.connectionStatus == ConnectionStatus.connected;
        final isExecuting = state.isExecutingCommand;
        final isDisabled = !isConnected || isExecuting;

        return Card(
          elevation: 4,
          margin: const EdgeInsets.all(8),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildStatusBar(context, state),
                const SizedBox(height: 20),
                _buildBasicControls(context, state, isDisabled),
                const SizedBox(height: 24),
                _buildMovementControls(context, state, isDisabled),
                const SizedBox(height: 24),
                _buildFlipControls(context, state, isDisabled),
                const SizedBox(height: 24),
                _buildRotationControls(context, state, isDisabled),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildStatusBar(BuildContext context, DroneState state) {
    String statusText;
    Color statusColor;
    
    switch (state.connectionStatus) {
      case ConnectionStatus.connected:
        statusText = 'Connected';
        statusColor = Colors.green;
        break;
      case ConnectionStatus.connecting:
        statusText = 'Connecting...';
        statusColor = Colors.orange;
        break;
      case ConnectionStatus.disconnected:
        statusText = 'Disconnected';
        statusColor = Colors.red;
        break;
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            Icon(Icons.circle, color: statusColor, size: 12),
            const SizedBox(width: 8),
            Text(statusText, style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        if (state.batteryLevel != null)
          Row(
            children: [
              Icon(
                state.batteryLevel! > 20 
                  ? Icons.battery_full 
                  : Icons.battery_alert,
                color: state.batteryLevel! > 20 ? Colors.green : Colors.red,
              ),
              const SizedBox(width: 4),
              Text('${state.batteryLevel}%'),
            ],
          ),
        ElevatedButton(
          onPressed: () {
            context.read<DroneBloc>().add(RequestBatteryLevelEvent());
          },
          child: const Text('Check Battery'),
        ),
      ],
    );
  }

  Widget _buildBasicControls(BuildContext context, DroneState state, bool isDisabled) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Basic Controls', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton.icon(
              onPressed: isDisabled ? null : () {
                context.read<DroneBloc>().add(
                  const SendCommandEvent(TakeoffCommand())
                );
              },
              icon: const Icon(Icons.flight_takeoff),
              label: const Text('Take Off'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
              ),
            ),
            ElevatedButton.icon(
              onPressed: isDisabled ? null : () {
                context.read<DroneBloc>().add(
                  const SendCommandEvent(LandCommand())
                );
              },
              icon: const Icon(Icons.flight_land),
              label: const Text('Land'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                foregroundColor: Colors.white,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMovementControls(BuildContext context, DroneState state, bool isDisabled) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Movement Controls', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildDirectionButton(
              context,
              icon: Icons.arrow_upward,
              direction: 'forward',
              isDisabled: isDisabled,
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildDirectionButton(
              context, 
              icon: Icons.arrow_back,
              direction: 'left',
              isDisabled: isDisabled,
            ),
            const SizedBox(width: 16),
            _buildDirectionButton(
              context,
              icon: Icons.arrow_downward,
              direction: 'back',
              isDisabled: isDisabled,
            ),
            const SizedBox(width: 16),
            _buildDirectionButton(
              context,
              icon: Icons.arrow_forward,
              direction: 'right',
              isDisabled: isDisabled,
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildFlipControls(BuildContext context, DroneState state, bool isDisabled) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Flip Controls', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _buildFlipButton(context, 'l', 'Left', isDisabled),
            _buildFlipButton(context, 'r', 'Right', isDisabled),
            _buildFlipButton(context, 'f', 'Forward', isDisabled),
            _buildFlipButton(context, 'b', 'Back', isDisabled),
          ],
        ),
      ],
    );
  }

  Widget _buildRotationControls(BuildContext context, DroneState state, bool isDisabled) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Rotation Controls', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _buildRotationButton(context, 'ccw', MdiIcons.rotateLeft, 'Counter-Clockwise', isDisabled),
            const SizedBox(width: 24),
            _buildRotationButton(context, 'cw', MdiIcons.rotateRight, 'Clockwise', isDisabled),
          ],
        ),
      ],
    );
  }

  Widget _buildDirectionButton(
    BuildContext context, {
    required IconData icon,
    required String direction,
    required bool isDisabled,
  }) {
    return ElevatedButton(
      onPressed: isDisabled ? null : () {
        _showDistanceDialog(context, direction);
      },
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.all(16),
        shape: const CircleBorder(),
      ),
      child: Icon(icon),
    );
  }

  Widget _buildFlipButton(
    BuildContext context,
    String direction,
    String label,
    bool isDisabled,
  ) {
    return ElevatedButton(
      onPressed: isDisabled ? null : () {
        context.read<DroneBloc>().add(
          SendCommandEvent(FlipCommand(direction))
        );
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.purple,
        foregroundColor: Colors.white,
      ),
      child: Text('Flip $label'),
    );
  }

  Widget _buildRotationButton(
    BuildContext context,
    String direction,
    IconData icon,
    String label,
    bool isDisabled,
  ) {
    return ElevatedButton.icon(
      onPressed: isDisabled ? null : () {
        _showDegreeDialog(context, direction);
      },
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.teal,
        foregroundColor: Colors.white,
      ),
    );
  }

  void _showDistanceDialog(BuildContext context, String direction) {
    final TextEditingController controller = TextEditingController(text: '20');
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Move ${direction.capitalize()} Distance'),
        content: TextField(
          controller: controller,
          keyboardType: TextInputType.number,
          decoration: const InputDecoration(
            labelText: 'Distance (cm)',
            hintText: 'Enter a value between 20-500',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final distance = int.tryParse(controller.text) ?? 20;
              if (distance >= 20 && distance <= 500) {
                context.read<DroneBloc>().add(
                  SendCommandEvent(MoveCommand(direction, distance))
                );
                Navigator.of(context).pop();
              }
            },
            child: const Text('Move'),
          ),
        ],
      ),
    );
  }

  void _showDegreeDialog(BuildContext context, String direction) {
    final TextEditingController controller = TextEditingController(text: '90');
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Rotate ${direction == 'cw' ? 'Clockwise' : 'Counter-Clockwise'}'),
        content: TextField(
          controller: controller,
          keyboardType: TextInputType.number,
          decoration: const InputDecoration(
            labelText: 'Degrees',
            hintText: 'Enter a value between 1-360',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final degrees = int.tryParse(controller.text) ?? 90;
              if (degrees >= 1 && degrees <= 360) {
                context.read<DroneBloc>().add(
                  SendCommandEvent(RotateCommand(direction, degrees))
                );
                Navigator.of(context).pop();
              }
            },
            child: const Text('Rotate'),
          ),
        ],
      ),
    );
  }
}

extension StringExtension on String {
  String capitalize() {
    return "${this[0].toUpperCase()}${substring(1)}";
  }
}