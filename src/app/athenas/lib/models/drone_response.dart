class DroneResponse {
  final String command;
  final String status;
  final dynamic response;
  final String? message;

  DroneResponse({
    required this.command,
    required this.status,
    this.response,
    this.message,
  });

  factory DroneResponse.fromJson(Map<String, dynamic> json) {
    // Melhorar o tratamento para diferentes formatos de resposta
    String message = '';
    
    // Processar o campo 'message' que pode vir em diferentes formatos
    if (json['message'] != null) {
      message = json['message'].toString();
    } else if (json['response'] != null && json['response'] is String) {
      message = json['response'];
    } else if (json['data'] != null) {
      message = json['data'].toString();
    }
    
    return DroneResponse(
      command: json['command'] ?? '',
      status: json['status'] ?? 'ok', // Assume 'ok' se não for especificado
      response: json['response'],
      message: message,
    );
  }

  bool get isSuccess => status == 'ok';
}