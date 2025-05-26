import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import placeholder from '../assets/placeholder-icon.svg';
import { Trash2 } from 'lucide-react';

const DroneImages = () => {
  const [searchParams] = useSearchParams();
  const projetoAtivo = searchParams.get("projeto");

  const [selectedImage, setSelectedImage] = useState(null);

  const [images, setImages] = useState([
    { id: 1, src: placeholder, enviado: false, enviadoPor: '', nome: 'Captura 1', projeto: 'usp' },
    { id: 2, src: placeholder, enviado: true, enviadoPor: 'Especialista 1', nome: 'Captura 2', projeto: 'usp' },
    { id: 3, src: placeholder, enviado: false, enviadoPor: '', nome: 'Captura 3', projeto: 'meta' },
    { id: 4, src: placeholder, enviado: false, enviadoPor: '', nome: 'Captura 4', projeto: 'meta' },
  ]);

  const imagensDoProjeto = images.filter(img => img.projeto === projetoAtivo);

  const handleEnviarParaModelo = () => {
    const nomeUsuario = 'Especialista 1';
    setImages((prev) =>
      prev.map((img) =>
        img.id === selectedImage.id
          ? { ...img, enviado: true, enviadoPor: nomeUsuario }
          : img
      )
    );
    setSelectedImage(null);
  };

  const handleDeleteImage = (id) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Imagem atual do drone */}
      <div className="mb-10 text-center">
        <label className="block text-3xl font-semibold mb-4">Imagem do Drone</label>
        <div className="mx-auto bg-gray-200 w-[50%] p-14 h-[350px] rounded-md flex justify-center items-center">
          <img
            src={placeholder}
            alt="Imagem do drone"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <button
          className="mt-6 bg-blue-600 hover:bg-blue-700 text-white text-lg font-medium px-8 py-3 rounded-lg shadow"
          onClick={() => alert('Capturar imagem ainda não implementado')}
        >
          CAPTURAR IMAGEM
        </button>
      </div>

      {/* Galeria de imagens capturadas */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Imagens Capturadas</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {imagensDoProjeto.map((imagem) => (
            <div key={imagem.id} className="relative border rounded-lg p-4 bg-white shadow">
              {/* Status */}
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

              {/* Imagem */}
              <img
                src={imagem.src}
                alt={imagem.nome}
                className="w-full h-48 object-contain rounded cursor-pointer"
                onClick={() => setSelectedImage(imagem)}
              />

              {/* Lixeira */}
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
      </div>

      {/* Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white rounded p-6 max-w-md w-full shadow-lg">
            <h2 className="text-xl font-semibold mb-4">{selectedImage.nome}</h2>
            <img
              src={selectedImage.src}
              alt={selectedImage.nome}
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

export default DroneImages;