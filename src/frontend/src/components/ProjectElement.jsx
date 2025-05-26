"use client";

import placeholder from '../assets/placeholder-icon.svg';

const ProjectElement = ({ name, onViewReport }) => {
  return (
    <div className="bg-white border border-gray-300 rounded-md shadow-md p-6 flex justify-between items-center">
      <div className="flex items-center gap-4">
        <div className="w-20 h-20 bg-gray-100 flex items-center justify-center rounded-md">
          <img src={placeholder} alt="Imagem do projeto" className="max-h-full max-w-full object-contain" />
        </div>
        <h3 className="text-xl font-semibold text-dark-blue">{name}</h3>
      </div>
      <button
        onClick={() => onViewReport(name)}
        className="bg-dark-blue text-white px-4 py-2 rounded-md hover:bg-blue-darker transition"
      >
        VER MAIS
      </button>
    </div>
  );
};

export default ProjectElement;
