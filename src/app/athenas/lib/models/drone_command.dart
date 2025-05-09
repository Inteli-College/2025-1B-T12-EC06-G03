import 'package:equatable/equatable.dart';

abstract class DroneCommand extends Equatable {
  final String command;
  
  const DroneCommand(this.command);
  
  Map<String, dynamic> toJson();
  
  @override
  List<Object?> get props => [command];
}

class TakeoffCommand extends DroneCommand {
  const TakeoffCommand() : super('takeoff');
  
  @override
  Map<String, dynamic> toJson() => {};
}

class LandCommand extends DroneCommand {
  const LandCommand() : super('land');
  
  @override
  Map<String, dynamic> toJson() => {};
}

class FlipCommand extends DroneCommand {
  final String direction;
  
  const FlipCommand(this.direction) : super('flip');
  
  @override
  Map<String, dynamic> toJson() => {'direction': direction};
  
  @override
  List<Object?> get props => [...super.props, direction];
}

class MoveCommand extends DroneCommand {
  final String direction;
  final int distance;
  
  const MoveCommand(this.direction, this.distance) : super('move');
  
  @override
  Map<String, dynamic> toJson() => {
    'direction': direction,
    'distance': distance,
  };
  
  @override
  List<Object?> get props => [...super.props, direction, distance];
}

class RotateCommand extends DroneCommand {
  final String direction;
  final int degree;
  
  const RotateCommand(this.direction, this.degree) : super('rotate');
  
  @override
  Map<String, dynamic> toJson() => {
    'direction': direction,
    'degree': degree,
  };
  
  @override
  List<Object?> get props => [...super.props, direction, degree];
}

class BatteryCommand extends DroneCommand {
  const BatteryCommand() : super('battery');
  
  @override
  Map<String, dynamic> toJson() => {};
}