class ServerConfig {
  final String host;
  final int port;
  final String? savePath;  // Caminho para salvar vídeos e fotos

  const ServerConfig({
    required this.host,
    required this.port,
    this.savePath,
  });

  factory ServerConfig.defaultConfig() {
    return const ServerConfig(
      host: '10.32.0.11',
      port: 5000,
      savePath: null,  // Será definido dinamicamente para o diretório de documentos do usuário
    );
  }

  String get serverUrl => 'http://$host:$port';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ServerConfig && 
           other.host == host && 
           other.port == port &&
           other.savePath == savePath;
  }

  @override
  int get hashCode => host.hashCode ^ port.hashCode ^ (savePath?.hashCode ?? 0);
}