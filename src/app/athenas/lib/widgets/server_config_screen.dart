import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
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
  late TextEditingController _savePathController;
  
  @override
  void initState() {
    super.initState();
    
    // Initialize with current config from BLoC
    final currentConfig = context.read<DroneBloc>().state.serverConfig;
    _hostController = TextEditingController(text: currentConfig.host);
    _portController = TextEditingController(text: currentConfig.port.toString());
    _savePathController = TextEditingController(text: currentConfig.savePath ?? '');
    
    // Se não houver caminho configurado, obtenha o diretório padrão de documentos
    if (_savePathController.text.isEmpty) {
      _getDefaultDocumentsPath();
    }
  }
  
  Future<void> _getDefaultDocumentsPath() async {
    try {
      final documentsDir = await getApplicationDocumentsDirectory();
      setState(() {
        _savePathController.text = documentsDir.path;
      });
    } catch (e) {
      // Se não conseguir obter o diretório padrão, deixe vazio
      print('Erro ao obter diretório de documentos: $e');
    }
  }
  
  Future<void> _pickDirectory() async {
    try {
      String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
      
      if (selectedDirectory != null) {
        setState(() {
          _savePathController.text = selectedDirectory;
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Erro ao selecionar diretório: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  void dispose() {
    _hostController.dispose();
    _portController.dispose();
    _savePathController.dispose();
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
            const SizedBox(height: 8),
            Divider(),
            const SizedBox(height: 4),
            _buildConfigInfoRow(
              Icons.save, 
              'Diretório de Salvamento: ${config.savePath ?? "Não configurado"}',
              fontSize - 2,
            ),
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
        SizedBox(height: spacing),
        Text(
          'Configuração de Salvamento',
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: spacing * 0.5),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: TextFormField(
                controller: _savePathController,
                decoration: const InputDecoration(
                  labelText: 'Diretório para Salvar Fotos e Vídeos',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.folder),
                ),
                style: TextStyle(fontSize: fontSize - 2),
                readOnly: true,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Por favor, selecione um diretório para salvamento';
                  }
                  return null;
                },
              ),
            ),
            SizedBox(width: spacing * 0.5),
            ElevatedButton(
              onPressed: _pickDirectory,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.all(spacing * 0.8),
                minimumSize: Size(spacing * 4, 56),
              ),
              child: const Icon(Icons.folder_open, size: 36),
            ),
          ],
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
          final savePath = _savePathController.text.trim();
          
          final newConfig = ServerConfig(
            host: host, 
            port: port, 
            savePath: savePath,
          );
          
          // Update configuration via BLoC
          context.read<DroneBloc>().add(UpdateServerConfigEvent(newConfig));
          
          // Show a snackbar to indicate the change
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Configuração atualizada:'),
                  Text('Servidor: ${newConfig.serverUrl}'),
                  Text('Diretório de salvamento: ${savePath}'),
                ],
              ),
              backgroundColor: Colors.green,
              duration: const Duration(seconds: 3),
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