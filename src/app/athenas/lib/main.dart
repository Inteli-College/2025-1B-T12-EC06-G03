import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'services/drone_service.dart';
import 'bloc/drone_bloc.dart';
import 'bloc/drone_event.dart';
import 'bloc/drone_state.dart';
import 'models/drone_command.dart';
import 'widgets/video_stream_widget.dart';
import 'widgets/status_notification.dart';
import 'widgets/server_config_screen.dart';
import 'widgets/drone_joystick_control.dart';
import 'widgets/altitude_joystick_control.dart';
import 'dart:math' as math;

void main() {
  runApp(const DroneControlApp());
}

class DroneControlApp extends StatelessWidget {
  const DroneControlApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Athenas Drone Control',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: BlocProvider(
        create: (context) => DroneBloc(DroneService()),
        child: const DroneControlScreen(),
      ),
    );
  }
}

class DroneControlScreen extends StatefulWidget {
  const DroneControlScreen({Key? key}) : super(key: key);

  @override
  State<DroneControlScreen> createState() => _DroneControlScreenState();
}

class _DroneControlScreenState extends State<DroneControlScreen> {
  @override
  Widget build(BuildContext context) {
    final droneBloc = context.watch<DroneBloc>();
    final videoUrl = droneBloc.getVideoStreamUrl();
    final state = droneBloc.state;

    final screenSize = MediaQuery.of(context).size;
    final isSmallScreen = screenSize.width < 600;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(
          'Athenas Drone Control',
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
          textScaleFactor: isSmallScreen ? 0.9 : 1.0,
        ),
        backgroundColor: Colors.black.withOpacity(0.5),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.battery_full, color: Colors.white),
            onPressed: () {
              droneBloc.add(RequestBatteryLevelEvent());
            },
          ),
          if (state.batteryLevel != null)
            Center(
              child: Padding(
                padding: EdgeInsets.only(right: isSmallScreen ? 4.0 : 8.0),
                child: Text(
                  '${state.batteryLevel}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white),
            tooltip: 'Server Settings',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) => BlocProvider.value(
                    value: droneBloc,
                    child: const ServerConfigScreen(),
                  ),
                ),
              );
            },
          ),
        ],
      ),
      body: _buildResponsiveLayout(context, videoUrl, state, screenSize),
    );
  }

  Widget _buildResponsiveLayout(BuildContext context, String videoUrl,
      DroneState state, Size screenSize) {
    final joystickSize = _getResponsiveJoystickSize(screenSize);
    final padding = _getResponsivePadding(screenSize);
    final isSmallScreen = screenSize.width < 600;

    final isConnected = state.connectionStatus == ConnectionStatus.connected;
    final isExecuting = state.isExecutingCommand;
    final isDisabled = !isConnected;

    return Stack(
      fit: StackFit.expand,
      children: [
        VideoStreamWidget(streamUrl: videoUrl),
        Positioned(
          top: kToolbarHeight + (isSmallScreen ? 5 : 10),
          left: 0,
          right: 0,
          child: Padding(
            padding: EdgeInsets.symmetric(horizontal: padding),
            child: _buildConnectionStatus(state, screenSize),
          ),
        ),
        Positioned(
          left: padding,
          bottom: padding,
          child: AltitudeJoystickControl(
            size: joystickSize,
          ),
        ),
        Positioned(
          right: padding,
          bottom: padding,
          child: DroneJoystickControl(
            size: joystickSize,
            label: 'Movimento',
          ),
        ),
        Positioned(
          top: kToolbarHeight + (isSmallScreen ? 35 : 50),
          right: padding,
          child: _buildActionButtons(context, isDisabled, screenSize),
        ),
        Positioned(
          bottom: joystickSize + padding * 2,
          left: 0,
          right: 0,
          child: StatusNotification(),
        ),
      ],
    );
  }

  double _getResponsiveJoystickSize(Size screenSize) {
    final smallestDimension = math.min(screenSize.width, screenSize.height);
    final joystickSize = smallestDimension * 0.22;
    return joystickSize.clamp(100.0, 150.0);
  }

  double _getResponsivePadding(Size screenSize) {
    final smallestDimension = math.min(screenSize.width, screenSize.height);
    final padding = smallestDimension * 0.03;
    return padding.clamp(10.0, 30.0);
  }

  Widget _buildConnectionStatus(DroneState state, Size screenSize) {
    String statusText;
    Color statusColor;

    switch (state.connectionStatus) {
      case ConnectionStatus.connected:
        statusText = 'Conectado';
        statusColor = Colors.green;
        break;
      case ConnectionStatus.connecting:
        statusText = 'Conectando...';
        statusColor = Colors.orange;
        break;
      case ConnectionStatus.disconnected:
        statusText = 'Desconectado';
        statusColor = Colors.red;
        break;
    }

    final fontSize = screenSize.width < 600 ? 12.0 : 14.0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 10,
            height: 10,
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            statusText,
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: fontSize,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(
      BuildContext context, bool isDisabled, Size screenSize) {
    final isSmallScreen = screenSize.width < 600;
    final buttonPadding = EdgeInsets.symmetric(
      horizontal: isSmallScreen ? 12 : 16,
      vertical: isSmallScreen ? 6 : 8,
    );
    final spacingHeight = isSmallScreen ? 6.0 : 10.0;
    final iconSize = isSmallScreen ? 18.0 : 24.0;
    final fontSize = isSmallScreen ? 12.0 : 14.0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        _buildActionButton(
          context,
          icon: Icons.flight_takeoff,
          label: 'Decolar',
          color: Colors.blue,
          onPressed: isDisabled
              ? null
              : () {
                  context
                      .read<DroneBloc>()
                      .add(const SendCommandEvent(TakeoffCommand()));
                },
          padding: buttonPadding,
          iconSize: iconSize,
          fontSize: fontSize,
        ),
        SizedBox(height: spacingHeight),
        _buildActionButton(
          context,
          icon: Icons.flight_land,
          label: 'Pousar',
          color: Colors.red,
          onPressed: isDisabled
              ? null
              : () {
                  context
                      .read<DroneBloc>()
                      .add(const SendCommandEvent(LandCommand()));
                },
          padding: buttonPadding,
          iconSize: iconSize,
          fontSize: fontSize,
        ),
        SizedBox(height: spacingHeight),
        _buildActionButton(
          context,
          icon: Icons.autorenew,
          label: 'Flip',
          color: Colors.purple,
          onPressed: isDisabled
              ? null
              : () {
                  _showFlipDirectionDialog(context, screenSize);
                },
          padding: buttonPadding,
          iconSize: iconSize,
          fontSize: fontSize,
        ),
      ],
    );
  }

  Widget _buildActionButton(
    BuildContext context, {
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback? onPressed,
    required EdgeInsets padding,
    required double iconSize,
    required double fontSize,
  }) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: iconSize),
      label: Text(
        label,
        style: TextStyle(fontSize: fontSize),
      ),
      style: ElevatedButton.styleFrom(
        backgroundColor: color.withOpacity(onPressed == null ? 0.5 : 0.8),
        foregroundColor: Colors.white,
        padding: padding,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
      ),
    );
  }

  void _showFlipDirectionDialog(BuildContext context, Size screenSize) {
    final isSmallScreen = screenSize.width < 600;
    final iconSize = isSmallScreen ? 24.0 : 30.0;
    final fontSize = isSmallScreen ? 12.0 : 14.0;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.black87,
        title: const Text(
          'Escolha a direção do flip',
          style: TextStyle(color: Colors.white),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildFlipButton(context, 'l', Icons.arrow_back, 'Esquerda',
                    iconSize, fontSize),
                _buildFlipButton(context, 'r', Icons.arrow_forward, 'Direita',
                    iconSize, fontSize),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildFlipButton(context, 'f', Icons.arrow_upward, 'Frente',
                    iconSize, fontSize),
                _buildFlipButton(context, 'b', Icons.arrow_downward, 'Trás',
                    iconSize, fontSize),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFlipButton(
    BuildContext context,
    String direction,
    IconData icon,
    String label,
    double iconSize,
    double fontSize,
  ) {
    return Column(
      children: [
        IconButton(
          onPressed: () {
            context
                .read<DroneBloc>()
                .add(SendCommandEvent(FlipCommand(direction)));
            Navigator.of(context).pop();
          },
          icon: Icon(icon, color: Colors.white, size: iconSize),
          style: IconButton.styleFrom(
            backgroundColor: Colors.purple.withOpacity(0.6),
            padding: const EdgeInsets.all(12),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(color: Colors.white, fontSize: fontSize),
        ),
      ],
    );
  }

  @override
  void dispose() {
    super.dispose();
  }
}
