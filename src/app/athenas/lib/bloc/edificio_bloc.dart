import 'package:athenas/models/edificio.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';
import 'package:dio/dio.dart';

// EVENTS
abstract class EdificioEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class FetchEdificios extends EdificioEvent {
  final int projectId;
  FetchEdificios(this.projectId);
  @override
  List<Object?> get props => [projectId];
}

// STATES
abstract class EdificioState extends Equatable {
  @override
  List<Object?> get props => [];
}

class EdificioInitial extends EdificioState {}
class EdificioLoading extends EdificioState {}
class EdificioLoaded extends EdificioState {
  final List<Edificio> edificios;
  EdificioLoaded(this.edificios);
  @override
  List<Object?> get props => [edificios];
}
class EdificioError extends EdificioState {
  final String message;
  EdificioError(this.message);
  @override
  List<Object?> get props => [message];
}

// BLOC
class EdificioBloc extends Bloc<EdificioEvent, EdificioState> {
  final Dio dio;
  EdificioBloc(this.dio) : super(EdificioInitial()) {
    on<FetchEdificios>(_onFetchEdificios);
  }

  Future<void> _onFetchEdificios(FetchEdificios event, Emitter<EdificioState> emit) async {
    emit(EdificioLoading());
    try {
      final response = await dio.get('/edificio/${event.projectId}/edificios');
      print('Response: ${response}');
      if (response.statusCode == 200 && response.data is List) {
        final edificios = (response.data as List)
            .map((json) => Edificio.fromJson(json as Map<String, dynamic>))
            .toList();
        emit(EdificioLoaded(edificios));
      } else {
        emit(EdificioError('Erro ao buscar edifícios'));
      }
    } catch (e) {
      emit(EdificioError('Erro: $e'));
    }
  }
}
