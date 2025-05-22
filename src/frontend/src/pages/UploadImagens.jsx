import { useState } from 'react';
import placeholder from '../assets/placeholder-icon.svg';
import { Trash2 } from 'lucide-react';

const UploadImagens = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  const [imagens, setImagens] = useState([
    {
      id: 1,
      src: placeholder,
      name: 'Placeholder 1',
      enviado: false,
      enviadoPor: '',
    },
    {
      id: 2,
      src: placeholder,
      name: 'Placeholder 2',
      enviado: true,
      enviadoPor: 'Especialista 1',
    },
  ]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagens((prevImagens) => [
          ...prevImagens,
          {
            id: Date.now(),
            src: e.target.result,
            name: file.name,
            enviado: false,
            enviadoPor: '',
          },
        ]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDeleteImage = (id) => {
    setImagens((prev) => prev.filter((img) => img.id !== id));
  };

  const handleEnviarParaModelo = () => {
    const nomeUsuario = 'Especialista 1'; // pode vir de login futuramente
    setImagens((prev) =>
      prev.map((img) =>
        img.id === selectedImage.id
          ? { ...img, enviado: true, enviadoPor: nomeUsuario }
          : img
      )
    );
    setSelectedImage(null);
  };

  return (
    <div className="min-h-screen bg-slate-100 p-8">
      <h1 className="text-3xl font-bold mb-6 text-dark-blue">Upload de Imagem</h1>

      {/* Área de Upload */}
      <div className="bg-gray-light h-72 flex items-center justify-center rounded-md mb-10">
        <label className="bg-dark-blue text-gray-light px-6 py-2 rounded-xl shadow-md cursor-pointer">
          Carregar Imagem
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </label>
      </div>

      <h2 className="text-3xl font-bold mb-6 text-dark-blue">Imagens Carregadas</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {imagens.map((imagem) => (
          <div key={imagem.id} className="relative border rounded-lg p-4 bg-white shadow">
            <div className="mb-2">
              {imagem.enviado ? (
                <span className="text-green-700 bg-green-100 px-2 py-1 rounded text-sm font-medium">
                  Mandado para o modelo por {imagem.enviadoPor}
                </span>
              ) : (
                <span className="text-yellow-800 bg-yellow-100 px-2 py-1 rounded text-sm font-medium">
                  Esperando ser mandado para o modelo
                </span>
              )}
            </div>
            <img
              src={imagem.src}
              alt={imagem.name}
              className="w-full h-48 object-contain rounded cursor-pointer"
              onClick={() => setSelectedImage(imagem)}
            />
            <button
              onClick={() => handleDeleteImage(imagem.id)}
              className="absolute top-2 right-2 text-red-500 hover:text-red-700"
              title="Deletar imagem"
            >
              <Trash2 size={20} />
            </button>
          </div>
        ))}
      </div>

      {/* Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white rounded p-6 max-w-md w-full shadow-lg">
            <h2 className="text-xl font-semibold mb-4">{selectedImage.name}</h2>
            <img
              src={selectedImage.src}
              alt={selectedImage.name}
              className="w-full h-auto mb-4 rounded"
            />
            {!selectedImage.enviado ? (
              <button
                onClick={handleEnviarParaModelo}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Mandar para o modelo
              </button>
            ) : (
              <p className="text-green-600 font-medium">
                Já mandado para o modelo por {selectedImage.enviadoPor}
              </p>
            )}
            <div className="mt-4 text-right">
              <button
                onClick={() => setSelectedImage(null)}
                className="text-sm text-gray-500 hover:underline"
              >
                Fechar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadImagens;
