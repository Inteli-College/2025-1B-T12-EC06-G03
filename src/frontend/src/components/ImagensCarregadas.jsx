import { useState } from 'react';
import placeholderIcon from '../assets/placeholder-icon.svg'; // Renomeado para evitar conflito

const ImagensCarregadas = ({ name, imgSrc }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleImageClick = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
    <div>
      {/* Adiciona padding interno ao contêiner azul */}
      <div className="bg-blue-dark rounded-md shadow-md p-5">
        <div
          className="relative bg-gray-light rounded-md overflow-hidden"
          style={{ paddingTop: '60%' }} // Mantém a proporção da imagem
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src={imgSrc || placeholderIcon} // Usa imgSrc ou o placeholder padrão
              alt={name || 'Imagem Placeholder'}
              className="w-3/4 h-3/4 object-contain cursor-pointer" // Ajusta o tamanho do placeholder
              onClick={handleImageClick} // Abre o modal ao clicar
            />
          </div>
        </div>
        <div className="flex justify-between items-center"></div>
      </div>

      {/* Modal para exibir a imagem ampliada */}
      {isModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={closeModal} // Fecha o modal ao clicar fora da imagem
        >
          <div className="relative">
            <img
              src={imgSrc || placeholderIcon}
              alt={name || 'Imagem Ampliada'}
              className="max-w-full max-h-full rounded-md"
            />
            <button
              className="absolute top-2 right-2 bg-white text-black rounded-full p-2"
              onClick={closeModal}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImagensCarregadas;
