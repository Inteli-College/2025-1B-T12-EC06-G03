import { useState } from 'react';
import placeholderIcon from '../assets/placeholder-icon.svg';

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
      <div className="bg-blue-dark rounded-md shadow-md p-2">
        <div
          className="relative bg-gray-light rounded-md overflow-hidden border border-gray-300"
          style={{ paddingTop: '50%' }} 
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src={imgSrc || placeholderIcon}
              alt={name || 'Imagem Placeholder'}
              className="w-full h-full object-contain cursor-pointer" 
              onClick={handleImageClick} 
            />
          </div>
        </div>
        <div className="flex justify-between items-center"></div>
      </div>
      <p className="text-center mt-2 text-[#010131]">{name}</p>

      {isModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={closeModal} 
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
