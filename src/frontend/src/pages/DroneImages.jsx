import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import placeholder from '../assets/placeholder-icon.svg';
import { Trash2 } from 'lucide-react';

const DroneImages = () => {
  const [searchParams] = useSearchParams();
  const projectIdParam = searchParams.get("id");
  const projectNameParam = searchParams.get("projeto");

  const [selectedImage, setSelectedImage] = useState(null);
  const [images, setImages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resolvedProjectId, setResolvedProjectId] = useState(null);

  // Supabase base URL for images
  const SUPABASE_BASE_URL = "https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/";

  // Function to resolve project ID from project name
  const resolveProjectId = async (projectName) => {
    try {
      const response = await fetch('http://localhost:8080/api/projetos');
      if (!response.ok) {
        throw new Error('Failed to fetch projects');
      }
      const projects = await response.json();
      const project = projects.find(p => p.nome.toLowerCase() === projectName.toLowerCase());
      return project ? project.id : null;
    } catch (err) {
      console.error('Error fetching projects:', err);
      return null;
    }
  };

  useEffect(() => {
    const initializeProjectId = async () => {
      let projectId = null;

      if (projectIdParam) {
        // If we have a direct project ID, use it
        projectId = projectIdParam;
      } else if (projectNameParam) {
        // If we only have a project name, resolve it to an ID
        projectId = await resolveProjectId(projectNameParam);
        if (!projectId) {
          setError(`Projeto "${projectNameParam}" não encontrado`);
          setIsLoading(false);
          return;
        }
      } else {
        setError('ID ou nome do projeto não fornecido');
        setIsLoading(false);
        return;
      }

      setResolvedProjectId(projectId);
    };

    initializeProjectId();
  }, [projectIdParam, projectNameParam]);

  useEffect(() => {
    const fetchImages = async () => {
      if (!resolvedProjectId) {
        return;
      }

      try {
        const response = await fetch(`http://localhost:8080/api/images/${resolvedProjectId}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Transform backend data to frontend format
        const transformedImages = result.map(image => ({
          id: image.id,
          src: SUPABASE_BASE_URL + image.caminhoArquivo,
          nome: image.nomeArquivo,
          enviado: image.processada, // Use processada as enviado status
          enviadoPor: image.processada ? 'Sistema' : '',
          projeto: resolvedProjectId,
          dataCaptura: image.dataCaptura,
          dataUpload: image.dataUpload,
          fachada: image.fachada?.nome || 'Sem fachada',
          edificio: image.fachada?.edificio?.nome || 'Sem edifício'
        }));
        
        setImages(transformedImages);
        setError(null);
      } catch (err) {
        setError(err.message || 'Failed to fetch images');
        console.error('Error fetching images:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchImages();
  }, [resolvedProjectId]);

  const handleEnviarParaModelo = () => {
    const nomeUsuario = localStorage.getItem('userName') || 'Usuário Atual';
    setImages((prev) =>
      prev.map((img) =>
        img.id === selectedImage.id
          ? { ...img, enviado: true, enviadoPor: nomeUsuario }
          : img
      )
    );
    setSelectedImage(null);
    // TODO: Implement API call to mark image as processed
    console.warn('Image marked as sent locally. Backend integration needed for persistence.');
  };

  const handleDeleteImage = (id) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
    // TODO: Implement API call to delete image from backend
    console.warn('Image deleted locally. Backend integration needed for persistence.');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg">Carregando imagens...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Erro:</strong> {error}
        </div>
      </div>
    );
  }

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
        {images.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            Nenhuma imagem encontrada para este projeto.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {images.map((imagem) => (
              <div key={imagem.id} className="relative border rounded-lg p-4 bg-white shadow">
                {/* Status */}
                <div className="mb-2">
                  {imagem.enviado ? (
                    <span className="text-green-700 bg-green-100 px-2 py-1 rounded text-sm font-medium">
                      Processada {imagem.enviadoPor && `por ${imagem.enviadoPor}`}
                    </span>
                  ) : (
                    <span className="text-yellow-800 bg-yellow-100 px-2 py-1 rounded text-sm font-medium">
                      Aguardando processamento
                    </span>
                  )}
                </div>

                {/* Additional info */}
                <div className="mb-2 text-xs text-gray-600">
                  <div>Fachada: {imagem.fachada}</div>
                  <div>Edifício: {imagem.edificio}</div>
                </div>

                {/* Imagem */}
                <img
                  src={imagem.src}
                  alt={imagem.nome}
                  className="w-full h-48 object-contain rounded cursor-pointer"
                  onClick={() => setSelectedImage(imagem)}
                  onError={(e) => {
                    e.target.src = placeholder;
                  }}
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
        )}
      </div>

      {/* Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
          <div className="bg-white rounded p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-lg">
            <h2 className="text-xl font-semibold mb-4">{selectedImage.nome}</h2>
            
            {/* Image details */}
            <div className="mb-4 text-sm text-gray-600">
              <div><strong>Fachada:</strong> {selectedImage.fachada}</div>
              <div><strong>Edifício:</strong> {selectedImage.edificio}</div>
              {selectedImage.dataCaptura && (
                <div><strong>Data de Captura:</strong> {new Date(selectedImage.dataCaptura).toLocaleDateString()}</div>
              )}
              <div><strong>Data de Upload:</strong> {new Date(selectedImage.dataUpload).toLocaleDateString()}</div>
            </div>

            {/* Image container with fixed height */}
            <div className="w-full h-80 mb-4 bg-gray-100 rounded flex items-center justify-center overflow-hidden">
              <img
                src={selectedImage.src}
                alt={selectedImage.nome}
                className="max-w-full max-h-full object-contain"
                onError={(e) => {
                  e.target.src = placeholder;
                }}
              />
            </div>
            
            <div className="flex justify-between items-center">
              {!selectedImage.enviado ? (
                <button
                  onClick={handleEnviarParaModelo}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Enviar para o modelo
                </button>
              ) : (
                <p className="text-green-600 font-medium">
                  Já processada {selectedImage.enviadoPor && `por ${selectedImage.enviadoPor}`}
                </p>
              )}
              <button
                onClick={() => setSelectedImage(null)}
                className="text-sm text-gray-500 hover:underline ml-4"
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