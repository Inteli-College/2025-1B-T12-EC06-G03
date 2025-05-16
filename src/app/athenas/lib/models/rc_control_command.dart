import 'package:equatable/equatable.dart';

class RCControlCommand extends Equatable {
  final Map<String, int> params;
  
  const RCControlCommand(this.params);
  
  @override
  List<Object?> get props => [params];
  
  Map<String, dynamic> toJson() => params;

  @override
  String toString() => 'RCControlCommand(lr: ${params['lr']}, fb: ${params['fb']}, ud: ${params['ud']}, yw: ${params['yw']})';
}
