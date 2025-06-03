import io from 'socket.io-client';

class ImageInferenceService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
  }

  connect() {
    if (this.socket && this.isConnected) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      this.socket = io('http://localhost:5000/ws/infer', {
        transports: ['websocket'],
        timeout: 30000,
      });

      this.socket.on('connect', () => {
        console.log('Conectado ao servidor de inferência');
        this.isConnected = true;
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        console.error('Erro de conexão:', error);
        this.isConnected = false;
        reject(error);
      });

      this.socket.on('disconnect', () => {
        console.log('Desconectado do servidor de inferência');
        this.isConnected = false;
      });
    });
  }

  inferImages(imageIds, callbacks = {}) {
    if (!this.socket || !this.isConnected) {
      throw new Error('Socket não conectado');
    }

    return new Promise((resolve, reject) => {
      const {
        onStatus = () => {},
        onResults = () => {},
        onError = () => {},
        onComplete = () => {}
      } = callbacks;

      // Listeners para os eventos do backend
      this.socket.once('status', (data) => {
        console.log('Status:', data.message);
        onStatus(data);
      });

      this.socket.once('results', (data) => {
        console.log('Resultados recebidos:', data.results);
        onResults(data.results);
      });

      this.socket.once('fim', (data) => {
        console.log('Processamento completo:', data.message);
        onComplete(data);
        resolve(data);
      });

      this.socket.once('error', (data) => {
        console.error('Erro no processamento:', data.error);
        onError(data);
        reject(new Error(data.error));
      });

      // Enviar imagens para processamento
      this.socket.emit('infer_images', { image_ids: imageIds });
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }
}

// Função para obter informações do usuário logado
export const getCurrentUser = async () => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      return null;
    }

    const response = await fetch('http://localhost:8080/auth/@me', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (response.ok) {
      const user = await response.json();
      return user;
    }
  } catch (error) {
    console.error('Erro ao obter usuário:', error);
  }
  return null;
};

// Singleton instance
const imageInferenceService = new ImageInferenceService();

export default imageInferenceService;