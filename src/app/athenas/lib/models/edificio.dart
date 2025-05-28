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
      projeto: json['projeto'] as Project?,
      nome: json['nome'] as String,
      localizacao: json['localizacao'] as String,
      tipo: json['tipo'] as String,
      pavimentos: json['pavimentos'] as int,
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