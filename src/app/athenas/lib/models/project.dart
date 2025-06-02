
class Project {
  final int? id;
  final String name;
  final String? description;
  final DateTime? createdAt;
  final DateTime? updatedAt;
  final String? status;

  Project({
    this.id,
    required this.name,
    this.description,
    this.createdAt,
    this.updatedAt,
    this.status,
  });

  factory Project.fromJson(Map<String, dynamic> json) {
    return Project(
      id: json['id'] as int?,
      name: json['nome'] as String,
      description: json['descricao'] as String?,
      createdAt: json['data_criacao'] != null ? DateTime.parse(json['data_criacao']) : null,
      updatedAt: json['data_atualizacao'] != null ? DateTime.parse(json['data_atualizacao']) : null,
      status: json['status'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'nome': name,
      'descricao': description,
      'data_criacao': createdAt?.toIso8601String(),
      'data_atualizacao': updatedAt?.toIso8601String(),
      'status': status,
    };
  }
}