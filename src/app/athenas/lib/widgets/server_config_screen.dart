import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../bloc/drone_bloc.dart';
import '../bloc/drone_event.dart';
import '../bloc/drone_state.dart';
import '../models/server_config.dart';

class ServerConfigScreen extends StatefulWidget {
  const ServerConfigScreen({Key? key}) : super(key: key);

  @override
  State<ServerConfigScreen> createState() => _ServerConfigScreenState();
}

class _ServerConfigScreenState extends State<ServerConfigScreen> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _hostController;
  late TextEditingController _portController;

  @override
  void initState() {
    super.initState();
    
    // Initialize with current config from BLoC
    final currentConfig = context.read<DroneBloc>().state.serverConfig;
    _hostController = TextEditingController(text: currentConfig.host);
    _portController = TextEditingController(text: currentConfig.port.toString());
  }

  @override
  void dispose() {
    _hostController.dispose();
    _portController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Obtém o tamanho da tela para ajustar o layout
    final screenSize = MediaQuery.of(context).size;
    final isSmallScreen = screenSize.width < 600;
    final padding = isSmallScreen ? 12.0 : 16.0;
    final fontSize = isSmallScreen ? 14.0 : 16.0;
    final spacing = isSmallScreen ? 12.0 : 16.0;
    
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Configuração do Servidor',
          style: TextStyle(fontSize: isSmallScreen ? 18.0 : 20.0),
        ),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: BlocBuilder<DroneBloc, DroneState>(
        builder: (context, state) {
          return Padding(
            padding: EdgeInsets.all(padding),
            child: Form(
              key: _formKey,
              child: SingleChildScrollView(
                child: ConstrainedBox(
                  constraints: BoxConstraints(
                    // Garante uma altura mínima igual à altura da tela menos a altura da AppBar
                    minHeight: screenSize.height - kToolbarHeight - MediaQuery.of(context).padding.top,
                  ),
                  child: IntrinsicHeight(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // Status de conexão
                        _buildConnectionStatus(state, fontSize),
                        SizedBox(height: spacing * 1.5),
                        
                        // Configuração atual
                        _buildCurrentConfig(state.serverConfig, fontSize),
                        SizedBox(height: spacing * 2),
                        
                        // Formulário
                        _buildConfigForm(fontSize, spacing),
                        SizedBox(height: spacing * 1.5),
                        
                        // Botões de ação
                        _buildActionButtons(context, state, isSmallScreen),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildConnectionStatus(DroneState state, double fontSize) {
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

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: statusColor),
      ),
      child: Row(
        children: [
          Icon(Icons.circle, color: statusColor, size: fontSize),
          const SizedBox(width: 8),
          Text(
            'Status: $statusText',
            style: TextStyle(
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
              color: statusColor,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCurrentConfig(ServerConfig config, double fontSize) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Configuração Atual',
              style: TextStyle(
                fontSize: fontSize,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            _buildConfigInfoRow(Icons.computer, 'Host: ${config.host}', fontSize - 2),
            const SizedBox(height: 4),
            _buildConfigInfoRow(Icons.router, 'Porta: ${config.port}', fontSize - 2),
            const SizedBox(height: 4),
            _buildConfigInfoRow(Icons.link, 'URL: ${config.serverUrl}', fontSize - 2),
          ],
        ),
      ),
    );
  }
  
  Widget _buildConfigInfoRow(IconData icon, String text, double fontSize) {
    return Row(
      children: [
        Icon(icon, size: fontSize + 4, color: Theme.of(context).primaryColor.withOpacity(0.7)),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(fontSize: fontSize),
          ),
        ),
      ],
    );
  }

  Widget _buildConfigForm(double fontSize, double spacing) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Nova Configuração',
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: spacing),
        TextFormField(
          controller: _hostController,
          decoration: const InputDecoration(
            labelText: 'Host',
            hintText: 'ex. 192.168.1.100 ou localhost',
            border: OutlineInputBorder(),
            prefixIcon: Icon(Icons.computer),
          ),
          style: TextStyle(fontSize: fontSize - 2),
          validator: (value) {
            if (value == null || value.isEmpty) {
              return 'Por favor, insira um endereço de host';
            }
            return null;
          },
        ),
        SizedBox(height: spacing),
        TextFormField(
          controller: _portController,
          decoration: const InputDecoration(
            labelText: 'Porta',
            hintText: 'ex. 3000',
            border: OutlineInputBorder(),
            prefixIcon: Icon(Icons.router),
          ),
          style: TextStyle(fontSize: fontSize - 2),
          keyboardType: TextInputType.number,
          inputFormatters: [
            FilteringTextInputFormatter.digitsOnly,
          ],
          validator: (value) {
            if (value == null || value.isEmpty) {
              return 'Por favor, insira um número de porta';
            }
            
            final port = int.tryParse(value);
            if (port == null) {
              return 'Por favor, insira um número válido';
            }
            
            if (port <= 0 || port > 65535) {
              return 'A porta deve estar entre 1 e 65535';
            }
            
            return null;
          },
        ),
      ],
    );
  }

  Widget _buildActionButtons(BuildContext context, DroneState state, bool isSmallScreen) {
    // Layout responsivo dos botões
    final buttonLayout = isSmallScreen 
        ? Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildCancelButton(context),
              const SizedBox(height: 8),
              _buildSaveButton(context),
            ],
          )
        : Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              _buildCancelButton(context),
              const SizedBox(width: 16),
              _buildSaveButton(context),
            ],
          );
    
    return buttonLayout;
  }
  
  Widget _buildCancelButton(BuildContext context) {
    return TextButton(
      onPressed: () {
        Navigator.of(context).pop();
      },
      style: TextButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      ),
      child: const Text('Cancelar'),
    );
  }
  
  Widget _buildSaveButton(BuildContext context) {
    return ElevatedButton(
      onPressed: () {
        if (_formKey.currentState!.validate()) {
          final host = _hostController.text.trim();
          final port = int.parse(_portController.text.trim());
          
          final newConfig = ServerConfig(host: host, port: port);
          
          // Update configuration via BLoC
          context.read<DroneBloc>().add(UpdateServerConfigEvent(newConfig));
          
          // Show a snackbar to indicate the change
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Configuração atualizada para ${newConfig.serverUrl}'),
              backgroundColor: Colors.green,
            ),
          );
          
          Navigator.of(context).pop();
        }
      },
      style: ElevatedButton.styleFrom(
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      ),
      child: const Text('Salvar e Conectar'),
    );
  }
}