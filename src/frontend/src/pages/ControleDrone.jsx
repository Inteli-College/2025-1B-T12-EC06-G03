import React, { useState, useEffect } from "react";
import { io } from "socket.io-client";

const ControleDrone = () => {
  // Constante para a URL do servidor
  const SERVER_URL = "10.32.0.11";
  const [logs, setLogs] = useState([]);
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Inicializar conexão Socket.IO
    const socketInstance = io(`http://${SERVER_URL}:5000`);
    
    socketInstance.on('connect', () => {
      appendLog('Conectado ao servidor Tello.');
      setIsConnected(true);
    });
    
    socketInstance.on('response', res => {
      appendLog(JSON.stringify(res));
    });
    
    socketInstance.on('connect_error', (err) => {
      appendLog(`Erro de conexão: ${err.message}`);
      setIsConnected(false);
    });
    
    socketInstance.on('disconnect', () => {
      appendLog('Desconectado do servidor Tello.');
      setIsConnected(false);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  const appendLog = (message) => {
    setLogs(prevLogs => [
      ...prevLogs, 
      { time: new Date().toLocaleTimeString(), message }
    ]);
  };

  const sendCommand = (command) => {
    if (socket && isConnected) {
      socket.emit(command);
      appendLog(`Comando enviado: ${command}`);
    } else {
      appendLog('Não foi possível enviar o comando: sem conexão');
    }
  };

  const move = (direction, distance) => {
    if (socket && isConnected) {
      socket.emit('move', { direction, distance });
      appendLog(`Comando enviado: move ${direction} ${distance}cm`);
    } else {
      appendLog('Não foi possível enviar o comando: sem conexão');
    }
  };

  const rotate = (direction, degree) => {
    if (socket && isConnected) {
      socket.emit('rotate', { direction, degree });
      appendLog(`Comando enviado: rotate ${direction} ${degree}°`);
    } else {
      appendLog('Não foi possível enviar o comando: sem conexão');
    }
  };

  const flip = (direction) => {
    if (socket && isConnected) {
      socket.emit('flip', { direction });
      appendLog(`Comando enviado: flip ${direction}`);
    } else {
      appendLog('Não foi possível enviar o comando: sem conexão');
    }
  };

  return (
    <div className="container mx-auto p-6 text-center">
      <h1 className="text-3xl font-bold text-dark-blue mb-6">Controle de Drone</h1>
      
      <div className="mb-8">
        <div className="relative overflow-hidden border-2 border-gray-800 rounded-lg bg-black w-full max-w-4xl mx-auto aspect-video">
          {isConnected ? (
            <img 
              src={`http://${SERVER_URL}:5000/video`} 
              alt="Vídeo Tello" 
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="flex items-center justify-center w-full h-full text-white text-lg p-4">
              Aguardando conexão com o drone...
            </div>
          )}
        </div>
      </div>

      <div className="mx-auto max-w-3xl mb-8">
        <div className="grid grid-cols-2 gap-2 mb-4">
          <button 
            onClick={() => sendCommand('takeoff')}
            className="bg-blue-darker hover:bg-dark-blue text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Decolar
          </button>
          <button 
            onClick={() => sendCommand('land')}
            className="bg-gray-medium hover:bg-gray-600 text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Pousar
          </button>
        </div>

        <div className="mb-4">
          <button 
            onClick={() => sendCommand('battery')}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition w-full"
          >
            Bateria
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          <button 
            onClick={() => move('forward', 50)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Frente
          </button>
          <button 
            onClick={() => move('back', 50)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Trás
          </button>
          <button 
            onClick={() => move('left', 50)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Esquerda
          </button>
          <button 
            onClick={() => move('right', 50)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Direita
          </button>
        </div>

        <div className="grid grid-cols-2 gap-2 mb-4">
          <button 
            onClick={() => rotate('cw', 90)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Girar CW
          </button>
          <button 
            onClick={() => rotate('ccw', 90)}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Girar CCW
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          <button 
            onClick={() => flip('l')}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Flip L
          </button>
          <button 
            onClick={() => flip('r')}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Flip R
          </button>
          <button 
            onClick={() => flip('f')}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Flip F
          </button>
          <button 
            onClick={() => flip('b')}
            className="bg-blue-dark hover:bg-blue-darker text-white font-semibold py-3 px-4 rounded-lg transition"
          >
            Flip B
          </button>
        </div>


      </div>

      <div className="mx-auto max-w-3xl">
        <h2 className="text-xl font-bold text-dark-blue mb-3">Log de respostas</h2>
        <div className="bg-gray-50 border border-gray-300 rounded-lg p-4 h-48 overflow-y-auto text-left">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className="mb-1">
                <span className="font-mono text-sm text-gray-500">{log.time}: </span>
                <span className="font-mono text-sm">{log.message}</span>
              </div>
            ))
          ) : (
            <p className="text-gray-500 italic">Nenhuma resposta recebida</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControleDrone;
