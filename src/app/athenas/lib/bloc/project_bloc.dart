import 'package:athenas/models/project.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:dio/dio.dart';

// EVENTS
abstract class ProjectEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class FetchProjects extends ProjectEvent {}

// STATES
abstract class ProjectState extends Equatable {
  @override
  List<Object?> get props => [];
}

class ProjectInitial extends ProjectState {}
class ProjectLoading extends ProjectState {}
class ProjectLoaded extends ProjectState {
  final List<Project> projects;
  ProjectLoaded(this.projects);
  @override
  List<Object?> get props => [projects];
}
class ProjectError extends ProjectState {
  final String message;
  ProjectError(this.message);
  @override
  List<Object?> get props => [message];
}

// BLOC
class ProjectBloc extends Bloc<ProjectEvent, ProjectState> {
  final Dio dio;
  ProjectBloc(this.dio) : super(ProjectInitial()) {
    on<FetchProjects>(_onFetchProjects);
  }

  Future<void> _onFetchProjects(FetchProjects event, Emitter<ProjectState> emit) async {
    emit(ProjectLoading());
    try {
      // Faz um GET para /projects esperando uma lista de projetos
      final response = await dio.get('/projetos');
      print('Response: ${response}');
      if (response.statusCode == 200 && response.data is List) {
        final projects = (response.data as List)
            .map((json) => Project.fromJson(json as Map<String, dynamic>))
            .toList();
        emit(ProjectLoaded(projects));
      } else {
        emit(ProjectError('Erro ao buscar projetos'));
      }
    } catch (e) {
      emit(ProjectError('Erro: $e'));
    }
  }
}
