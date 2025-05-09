class ServerConfig {
  final String host;
  final int port;

  const ServerConfig({
    required this.host,
    required this.port,
  });

  factory ServerConfig.defaultConfig() {
    return const ServerConfig(
      host: '10.32.0.11',
      port: 5000,
    );
  }

  String get serverUrl => 'http://$host:$port';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ServerConfig && other.host == host && other.port == port;
  }

  @override
  int get hashCode => host.hashCode ^ port.hashCode;
}