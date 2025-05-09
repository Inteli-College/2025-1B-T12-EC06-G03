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
    return DroneResponse(
      command: json['command'] ?? '',
      status: json['status'] ?? '',
      response: json['response'],
      message: json['message'],
    );
  }

  bool get isSuccess => status == 'ok';
}