import 'package:athenas/models/project.dart';

class Edificio {
  final int? id;
  final Project? projeto;
  final String nome;
  final String localizacao;
  final String tipo;
  final int pavimentos;

  Edificio({
    this.id,
    this.projeto,
    required this.nome,
    required this.localizacao,
    required this.tipo,
    required this.pavimentos,
  });

  factory Edificio.fromJson(Map<String, dynamic> json) {
    return Edificio(
      id: json['id'] as int?,
      projeto: json['projeto'] != null && json['projeto'] is Map<String, dynamic>
          ? Project.fromJson(json['projeto'] as Map<String, dynamic>)
          : null,
      nome: json['nome'] as String? ?? json['nome']?.toString() ?? '',
      localizacao: json['localizacao'] as String? ?? json['localizacao']?.toString() ?? '',
      tipo: json['tipo'] as String? ?? json['tipo']?.toString() ?? '',
      pavimentos: json['pavimentos'] is int
          ? json['pavimentos'] as int
          : int.tryParse(json['pavimentos']?.toString() ?? '') ?? 0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'projeto_id': projeto,
      'nome': nome,
      'localizacao': localizacao,
      'tipo': tipo,
      'pavimentos': pavimentos,
    };
  }
}