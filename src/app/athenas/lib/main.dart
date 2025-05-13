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
import 'widgets/splash_screen.dart';
import 'dart:math' as math;

void main() {
  runApp(const DroneControlApp());
}

class DroneControlApp extends StatelessWidget {
  const DroneControlApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => DroneBloc(DroneService()),
      child: MaterialApp(
        title: 'Athenas Drone Control',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          brightness: Brightness.dark,
          colorScheme: ColorScheme.dark(
            primary: Colors.deepPurple.shade300,
            secondary: Colors.deepPurple.shade200,
            surface: const Color(0xFF121212),
            background: const Color(0xFF121212),
          ),
          scaffoldBackgroundColor: const Color(0xFF121212),
          useMaterial3: true,
        ),
        home: const SplashScreen(
          nextScreen: DroneControlScreen(),
        ),
      ),
    );
  }
}

class DroneControlScreen extends StatefulWidget {
  const DroneControlScreen({Key? key}) : super(key: key);

  @override
  State<DroneControlScreen> createState() => _DroneControlScreenState();
}

class _DroneControlScreenState extends State<DroneControlScreen> with SingleTickerProviderStateMixin {
  // Global key for accessing the main joystick state from the altitude joystick
  final GlobalKey<DroneJoystickControlState> _mainJoystickKey = GlobalKey<DroneJoystickControlState>();
  
  // For action buttons animation
  late AnimationController _animationController;
  bool _showActionButtons = true;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );
    _animationController.value = 1.0; // Start with buttons visible
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }
  
  // Toggle the action buttons sidebar visibility
  void _toggleActionButtons() {
    setState(() {
      _showActionButtons = !_showActionButtons;
      if (_showActionButtons) {
        _animationController.forward();
      } else {
        _animationController.reverse();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final droneBloc = context.watch<DroneBloc>();
    final videoUrl = droneBloc.getVideoStreamUrl();
    final state = droneBloc.state;

    final screenSize = MediaQuery.of(context).size;
    // ignore: unused_local_variable
    final isSmallScreen = screenSize.width < 600; // Used for responsive layouts - preserved for future use

    return Scaffold(
      body: _buildResponsiveLayout(context, videoUrl, state, screenSize),
    );
  }

  Widget _buildResponsiveLayout(BuildContext context, String videoUrl,
      DroneState state, Size screenSize) {
    final joystickSize = _getResponsiveJoystickSize(screenSize);
    final padding = _getResponsivePadding(screenSize);
    // Used for responsive layout adjustments
    final isSmallScreen = screenSize.width < 600;

    final isConnected = state.connectionStatus == ConnectionStatus.connected;
    final isDisabled = !isConnected;

    return Stack(
      fit: StackFit.expand,
      children: [
        // Black background under video for more minimalist look
        Container(color: const Color(0xFF121212)),
        VideoStreamWidget(streamUrl: videoUrl),
        
        // Top menu bar with connection status, battery level, and settings
        Positioned(
          top: isSmallScreen ? 10 : 20,
          left: padding,
          right: padding,
          child: Row(
            children: [
              Expanded(
                child: _buildConnectionStatus(state, screenSize),
              ),
              if (state.batteryLevel != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.battery_full, 
                        color: Colors.deepPurple.shade200, 
                        size: 16
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${state.batteryLevel}%',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              const SizedBox(width: 8),
              IconButton(
                icon: Icon(Icons.settings, color: Colors.deepPurple.shade200),
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) => BlocProvider.value(
                        value: context.read<DroneBloc>(),
                        child: const ServerConfigScreen(),
                      ),
                    ),
                  );
                },
                style: IconButton.styleFrom(
                  backgroundColor: Colors.black54,
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: const Icon(Icons.battery_full, color: Colors.white),
                onPressed: () {
                  context.read<DroneBloc>().add(RequestBatteryLevelEvent());
                },
                style: IconButton.styleFrom(
                  backgroundColor: Colors.black54,
                ),
              ),
            ],
          ),
        ),
        
        // Joysticks
        Positioned(
          left: padding,
          bottom: padding,
          child: AltitudeJoystickControl(
            size: joystickSize,
            mainJoystickKey: _mainJoystickKey,
          ),
        ),
        Positioned(
          right: padding,
          bottom: padding,
          child: DroneJoystickControl(
            key: _mainJoystickKey,
            size: joystickSize,
          ),
        ),
        
        // Compact action buttons tray
        Positioned(
          right: padding,
          top: isSmallScreen ? 80 : 100,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              // Toggle button
              InkWell(
                onTap: _toggleActionButtons,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(15),
                    border: Border.all(
                      color: Colors.deepPurple.shade300.withOpacity(0.3),
                      width: 1,
                    ),
                  ),
                  child: Icon(
                    _showActionButtons ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                    color: Colors.deepPurple.shade200,
                    size: 24,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              // Action buttons tray
              AnimatedOpacity(
                duration: const Duration(milliseconds: 300),
                opacity: _showActionButtons ? 1.0 : 0.0,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  height: _showActionButtons ? null : 0,
                  width: _showActionButtons ? null : 0,
                  child: _showActionButtons
                      ? _buildActionButtons(context, isDisabled, screenSize)
                      : const SizedBox.shrink(),
                ),
              ),
            ],
          ),
        ),
        
        // Status notification
        Positioned(
          bottom: joystickSize + padding * 2,
          left: 0,
          right: 0,
          child: const StatusNotification(),
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
        statusColor = Colors.green.shade400;
        break;
      case ConnectionStatus.connecting:
        statusText = 'Conectando...';
        statusColor = Colors.orange.shade400;
        break;
      case ConnectionStatus.disconnected:
        statusText = 'Desconectado';
        statusColor = Colors.red.shade400;
        break;
    }

    final fontSize = screenSize.width < 600 ? 12.0 : 14.0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: statusColor.withOpacity(0.5),
          width: 1.0,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: statusColor,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: statusColor.withOpacity(0.5),
                  blurRadius: 4.0,
                  spreadRadius: 1.0,
                ),
              ],
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
    final iconSize = isSmallScreen ? 20.0 : 24.0;
    final containerSize = isSmallScreen ? 44.0 : 50.0;

    return Container(
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 10.0,
          ),
        ],
        border: Border.all(
          color: Colors.deepPurple.shade300.withOpacity(0.3),
          width: 1,
        ),
      ),
      padding: const EdgeInsets.all(6),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Takeoff button
          _buildCompactActionButton(
            context,
            icon: Icons.flight_takeoff,
            color: Colors.blue.shade700,
            onPressed: isDisabled
                ? null
                : () {
                    context
                        .read<DroneBloc>()
                        .add(const SendCommandEvent(TakeoffCommand()));
                  },
            iconSize: iconSize,
            size: containerSize,
            tooltip: 'Decolar',
          ),
          const SizedBox(height: 6),
          // Land button
          _buildCompactActionButton(
            context,
            icon: Icons.flight_land,
            color: Colors.red.shade700,
            onPressed: isDisabled
                ? null
                : () {
                    context
                        .read<DroneBloc>()
                        .add(const SendCommandEvent(LandCommand()));
                  },
            iconSize: iconSize,
            size: containerSize,
            tooltip: 'Pousar',
          ),
          const SizedBox(height: 6),
          // Flip button
          _buildCompactActionButton(
            context,
            icon: Icons.autorenew,
            color: Colors.deepPurple,
            onPressed: isDisabled
                ? null
                : () {
                    _showFlipDirectionDialog(context, screenSize);
                  },
            iconSize: iconSize,
            size: containerSize,
            tooltip: 'Flip',
          ),
        ],
      ),
    );
  }

  Widget _buildCompactActionButton(
    BuildContext context, {
    required IconData icon,
    required Color color,
    required VoidCallback? onPressed,
    required double iconSize,
    required double size,
    required String tooltip,
  }) {
    return SizedBox(
      width: size,
      child: Tooltip(
        message: tooltip,
        child: ElevatedButton(
          onPressed: onPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: color.withAlpha(((onPressed == null ? 0.3 : 0.7) * 255).toInt()),
            foregroundColor: Colors.white,
            elevation: onPressed == null ? 0 : 4,
            shadowColor: color.withAlpha((0.5 * 255).toInt()),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(10),
              side: BorderSide(
                color: onPressed == null
                    ? Colors.transparent
                    : color.withAlpha((0.2 * 255).toInt()),
                width: 1,
              ),
            ),
            padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: iconSize),
              const SizedBox(height: 4),
              Text(
                tooltip,
                style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w500),
                textAlign: TextAlign.center,
              ),
            ],
          ),
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
        backgroundColor: const Color(0xFF1E1E1E),
        surfaceTintColor: Colors.deepPurple.shade200,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(
            color: Colors.deepPurple.shade300.withOpacity(0.3),
            width: 1,
          ),
        ),
        title: Text(
          'Escolha a direção do flip',
          style: TextStyle(
            color: Colors.deepPurple.shade200,
            fontWeight: FontWeight.bold,
          ),
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
            // Debug print to verify direction value
            print('Flip button pressed with direction: $direction');
            
            // Create the command and log its properties
            final flipCommand = FlipCommand(direction);
            print('FlipCommand created: command=${flipCommand.command}, direction=${flipCommand.direction}');
            
            // Send the command to the bloc
            context
                .read<DroneBloc>()
                .add(SendCommandEvent(flipCommand));
                
            Navigator.of(context).pop();
          },
          icon: Icon(icon, color: Colors.white, size: iconSize),
          style: IconButton.styleFrom(
            backgroundColor: Colors.deepPurple.withOpacity(0.6),
            padding: const EdgeInsets.all(12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(15),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(color: Colors.white70, fontSize: fontSize),
        ),
      ],
    );
  }
}
