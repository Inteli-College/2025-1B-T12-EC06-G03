import 'dart:async';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'dart:ui' as ui;
import '../bloc/drone_bloc.dart';
import '../bloc/drone_state.dart';

class VideoStreamWidget extends StatefulWidget {
  final String? streamUrl;

  const VideoStreamWidget({
    Key? key,
    this.streamUrl,
  }) : super(key: key);

  @override
  State<VideoStreamWidget> createState() => _VideoStreamWidgetState();
}

class _VideoStreamWidgetState extends State<VideoStreamWidget> {
  Uint8List? _currentFrame;
  Uint8List? _lastFrame;
  bool _isConnecting = false;
  String? _errorMessage;
  StreamSubscription<List<int>>? _streamSubscription;
  StreamController<Uint8List>? _imageStreamController;
  
  // Buffer para acumular bytes do stream
  List<int> _buffer = [];

  // Variáveis para monitoramento de qualidade do stream
  int _framesReceived = 0;
  int _frameErrors = 0;
  DateTime? _lastFrameTime;
  Timer? _qualityMonitorTimer;
  Timer? _connectionTimeoutTimer;
  Duration _frameTimeout = const Duration(seconds: 5);

  // Variáveis para gravação de frames
  bool _isRecording = false;
  List<Uint8List> _recordedFrames = [];
  int _maxRecordedFrames = 100; // Limita a 100 frames (~3-4 segundos em 30fps)

  @override
  void initState() {
    super.initState();
    _connectToStream();
    // Iniciar monitoramento da qualidade do stream
    _startQualityMonitoring();
  }

  @override
  void didUpdateWidget(VideoStreamWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.streamUrl != widget.streamUrl) {
      _disconnectStream();
      _connectToStream();
    }
  }

  @override
  void dispose() {
    _disconnectStream();
    _imageStreamController?.close();
    _imageStreamController = null;
    _qualityMonitorTimer?.cancel();
    _connectionTimeoutTimer?.cancel();
    _recordedFrames.clear();
    super.dispose();
  }

  void _disconnectStream() {
    _streamSubscription?.cancel();
    _streamSubscription = null;
    _buffer.clear();
    _qualityMonitorTimer?.cancel();
    _connectionTimeoutTimer?.cancel();
    // Fechar o controller antigo, mas não o defina como nulo aqui
    // para evitar erros se alguma parte do código ainda estiver usando
    _imageStreamController?.close();
  }

  Future<void> _connectToStream() async {
    if (widget.streamUrl == null) {
      setState(() {
        _currentFrame = null;
        _isConnecting = false;
        _errorMessage = null;
      });
      return;
    }

    setState(() {
      _isConnecting = true;
      _errorMessage = null;
      _buffer.clear();
    });
    
    // Criar um novo StreamController com broadcast para permitir múltiplos ouvintes
    _imageStreamController?.close();
    _imageStreamController = StreamController<Uint8List>.broadcast();

    try {
      // Configurar cliente HTTP para processar o stream MJPEG
      final client = http.Client();
      final request = http.Request('GET', Uri.parse(widget.streamUrl!));
      
      // Adicionar o header ngrok-skip-browser-warning
      request.headers['ngrok-skip-browser-warning'] = 'true';
      
      final response = await client.send(request);

      if (response.statusCode != 200) {
        setState(() {
          _errorMessage = 'Erro ao conectar ao stream: ${response.statusCode}';
          _isConnecting = false;
        });
        return;
      }

      // Processar bytes do MJPEG stream
      final stream = response.stream;

      _streamSubscription = stream.listen(
        (List<int> newBytes) {
          // Adicionar novos bytes ao buffer
          _buffer.addAll(newBytes);
          
          // Processar frames completos
          _processFrames();
        },
        onError: (error) {
          setState(() {
            _errorMessage = 'Erro no stream: $error';
            _isConnecting = false;
          });
        },
        onDone: () {
          setState(() {
            _errorMessage = 'Conexão com o stream foi encerrada';
            _isConnecting = false;
          });
        },
      );

      // Escutar novos frames e atualizar UI
      _imageStreamController?.stream.listen((frameData) {
        if (mounted) {
          setState(() {
            _currentFrame = frameData;
            _lastFrame = frameData;
            _isConnecting = false;
          });
        }
      });

    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Erro de conexão: $e';
          _isConnecting = false;
        });
      }
    }
  }
  
  void _startQualityMonitoring() {
    _qualityMonitorTimer?.cancel();
    _qualityMonitorTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (_framesReceived > 0) {
        // Calcular taxa de erro e resetar contadores
        double errorRate = _frameErrors / (_framesReceived + _frameErrors);
        if (errorRate > 0.3 && mounted) {
          // Se a taxa de erro for alta, mostrar aviso
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Qualidade do stream baixa: ${(errorRate * 100).toStringAsFixed(0)}% de erros'),
              duration: const Duration(seconds: 3),
            ),
          );
        }
        _framesReceived = 0;
        _frameErrors = 0;
      }
    });
  }
  
  void _resetConnectionTimeout() {
    _connectionTimeoutTimer?.cancel();
    _connectionTimeoutTimer = Timer(_frameTimeout, () {
      if (mounted) {
        setState(() {
          _errorMessage = 'Timeout na conexão com o stream';
          _isConnecting = false;
        });
        // Tentar reconectar automaticamente
        _connectToStream();
      }
    });
  }

  void _processFrames() {
    // Reiniciar o timer de timeout sempre que novos bytes forem recebidos
    _resetConnectionTimeout();
    
    // Limitar o tamanho do buffer para evitar vazamento de memória
    final int maxBufferSize = 2 * 1024 * 1024; // 2MB limite
    if (_buffer.length > maxBufferSize) {
      _buffer = _buffer.sublist(_buffer.length - 100000); // Manter apenas os últimos 100KB
      _frameErrors++;
      return;
    }
    
    // Continuar processando enquanto houver bytes suficientes no buffer
    while (_buffer.length > 4) { // Precisamos de pelo menos alguns bytes para verificar marcadores
      // Procurar pelo início do frame JPEG (FF D8)
      int startMarkerIndex = _findSequence(_buffer, [0xFF, 0xD8], 0);
      
      if (startMarkerIndex == -1) {
        // Nenhum início de frame encontrado, manter apenas os últimos bytes para evitar perder um marcador dividido
        if (_buffer.length > 2) {
          _buffer = _buffer.sublist(_buffer.length - 2);
        }
        return;
      }
      
      // Remover bytes antes do início do frame
      if (startMarkerIndex > 0) {
        _buffer = _buffer.sublist(startMarkerIndex);
      }
      
      // Procurar pelo fim do frame JPEG (FF D9)
      int endMarkerIndex = _findSequence(_buffer, [0xFF, 0xD9], 2); // Começar depois do marcador inicial
      
      if (endMarkerIndex == -1) {
        // Frame incompleto, aguardar mais bytes
        return;
      }
      
      // Extrair o frame completo (incluindo o marcador final FF D9)
      final frameEndIndex = endMarkerIndex + 2;
      
      try {
        final frameBytes = Uint8List.fromList(_buffer.sublist(0, frameEndIndex));
        
        // Adicionar o frame ao stream se o controlador ainda estiver ativo e não nulo
        if (_imageStreamController != null && !_imageStreamController!.isClosed) {
          _imageStreamController!.add(frameBytes);
          _framesReceived++;
          _lastFrameTime = DateTime.now();
          _lastFrame = frameBytes;
          
          // Armazenar o frame se estiver gravando
          if (_isRecording) {
            _recordFrame(frameBytes);
          }
        }
      } catch (e) {
        // Log ou tratamento de erro se houver um problema ao processar o frame
        print('Erro ao processar frame: $e');
        _frameErrors++;
      }
      
      // Remover o frame processado do buffer
      _buffer = _buffer.sublist(frameEndIndex);
    }
  }
  
  // Método auxiliar para encontrar uma sequência de bytes no buffer
  int _findSequence(List<int> buffer, List<int> sequence, int startIndex) {
    if (buffer.length - startIndex < sequence.length) {
      return -1; // Buffer não tem tamanho suficiente para conter a sequência
    }
    
    for (int i = startIndex; i <= buffer.length - sequence.length; i++) {
      bool found = true;
      for (int j = 0; j < sequence.length; j++) {
        if (buffer[i + j] != sequence[j]) {
          found = false;
          break;
        }
      }
      if (found) {
        return i;
      }
    }
    
    return -1; // Sequência não encontrada
  }

  // Método para armazenar frames durante gravação
  void _recordFrame(Uint8List frameBytes) {
    _recordedFrames.add(frameBytes);
    // Limitar o número de frames para evitar consumo excessivo de memória
    if (_recordedFrames.length > _maxRecordedFrames) {
      _recordedFrames.removeAt(0);
    }
  }
  
  // Método para iniciar/parar gravação
  void toggleRecording() {
    setState(() {
      _isRecording = !_isRecording;
      
      if (!_isRecording && _recordedFrames.isNotEmpty) {
        // Se parou de gravar e tem frames, salvar
        _saveRecordedFrames();
      } else if (_isRecording) {
        // Se começou a gravar, limpar frames anteriores
        _recordedFrames.clear();
      }
    });
  }
  
  // Método para salvar os frames gravados
  Future<void> _saveRecordedFrames() async {
    if (_recordedFrames.isEmpty) return;

    try {
      // Usar caminho configurado ou o diretório de documentos como padrão
      final droneState = context.read<DroneBloc>().state;
      final configuredPath = droneState.serverConfig.savePath;
      
      String savePath;
      if (configuredPath != null && configuredPath.isNotEmpty) {
        savePath = configuredPath;
      } else {
        // Fallback para diretório de documentos se não houver configuração
        final directory = await getApplicationDocumentsDirectory();
        savePath = '${directory.path}/Documentos';
      }
      
      // Criar pasta com timestamp para este conjunto de frames
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final path = '$savePath/drone_capture_$timestamp';

      // Criar diretório se não existir
      await Directory(path).create(recursive: true);

      // Salvar cada frame como um arquivo JPEG
      for (int i = 0; i < _recordedFrames.length; i++) {
        final file = File('$path/frame_$i.jpg');
        await file.writeAsBytes(_recordedFrames[i]);
      }

      // Notificar usuário
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('${_recordedFrames.length} frames salvos em $path'),
            duration: const Duration(seconds: 3),
          ),
        );
      }

      _recordedFrames.clear();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Erro ao salvar frames: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    }
  }

  // Método para salvar o quadro atual
  Future<void> _saveCurrentFrame() async {
    if (_lastFrame == null) {
      return;
    }
    
    try {
      // Usar caminho configurado ou o diretório de documentos como padrão
      final droneState = context.read<DroneBloc>().state;
      final configuredPath = droneState.serverConfig.savePath;
      
      String savePath;
      if (configuredPath != null && configuredPath.isNotEmpty) {
        savePath = configuredPath;
      } else {
        // Fallback para diretório de documentos se não houver configuração
        final directory = await getApplicationDocumentsDirectory();
        savePath = directory.path;
      }
      
      // Criar pasta para fotos se não existir
      final photosDir = Directory('$savePath/drone_photos');
      if (!await photosDir.exists()) {
        await photosDir.create(recursive: true);
      }
      
      final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final fileName = 'drone_capture_$timestamp.jpg';
      final path = '${photosDir.path}/$fileName';
      
      // Salva a imagem
      final file = File(path);
      await file.writeAsBytes(_lastFrame!);
      
      // Notifica o usuário
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Imagem salva com sucesso!'),
              Text('Local: $path', style: const TextStyle(fontSize: 12)),
              Text('Nome: $fileName', style: const TextStyle(fontSize: 12)),
            ],
          ),
          backgroundColor: Colors.green,
          duration: const Duration(seconds: 3),
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Erro ao salvar imagem: $e'),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black,
      width: MediaQuery.of(context).size.width, // Força a largura total
      height: MediaQuery.of(context).size.height, // Força a altura total
      child: widget.streamUrl == null
          ? _buildNoStreamPlaceholder()
          : _buildVideoStream(),
    );
  }
  
  Widget _buildNoStreamPlaceholder() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.videocam_off,
            color: Colors.white.withOpacity(0.6),
            size: 80,
          ),
          const SizedBox(height: 20),
          Text(
            'Sem conexão com a câmera',
            style: TextStyle(
              color: Colors.white.withOpacity(0.8),
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            'Verifique as configurações do servidor',
            style: TextStyle(
              color: Colors.white.withOpacity(0.6),
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildVideoStream() {
    if (_isConnecting) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Colors.white),
            SizedBox(height: 20),
            Text(
              'Conectando ao stream...',
              style: TextStyle(color: Colors.white),
            ),
          ],
        ),
      );
    }

    if (_errorMessage != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              color: Colors.red.withOpacity(0.8),
              size: 70,
            ),
            const SizedBox(height: 20),
            Text(
              'Erro na transmissão',
              style: TextStyle(
                color: Colors.white.withOpacity(0.8),
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              _errorMessage!,
              style: TextStyle(
                color: Colors.white.withOpacity(0.6),
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                _disconnectStream();
                _connectToStream();
              },
              child: const Text('Tentar novamente'),
            ),
          ],
        ),
      );
    }

    return _currentFrame != null
        ? Stack(
            children: [
              Center(
                child: Image.memory(
                  _currentFrame!,
                  fit: BoxFit.cover,
                  width: double.infinity,
                  height: double.infinity,
                  gaplessPlayback: true,
                ),
              ),
                Align(
                alignment: Alignment.topLeft,
                child: Padding(
                  padding: const EdgeInsets.only(top: 80, left: 16),
                  child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    FloatingActionButton(
                    heroTag: 'camera_button',
                    backgroundColor: Colors.blue,
                    onPressed: _lastFrame != null ? _saveCurrentFrame : null,
                    child: const Icon(Icons.camera_alt, color: Colors.black),
                    ),
                    const SizedBox(height: 16),
                    FloatingActionButton(
                    heroTag: 'record_button',
                    backgroundColor: _isRecording ? Colors.red : Colors.blue,
                    onPressed: toggleRecording,
                    child: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record, color: Colors.black),
                    ),
                  ],
                  ),
                ),
                ),
            ],
          )
        : Center(
            child: Text(
              'Aguardando imagens do stream...',
              style: TextStyle(
                color: Colors.white.withOpacity(0.8),
                fontSize: 16,
              ),
            ),
          );
  }
}