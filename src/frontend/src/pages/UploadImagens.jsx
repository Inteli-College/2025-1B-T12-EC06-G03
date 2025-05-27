import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import placeholder from '../assets/placeholder-icon.svg';
import { Trash2 } from 'lucide-react';

const UploadImagens = () => {
  const [searchParams] = useSearchParams();
  const projectIdParam = searchParams.get("id");
  const projectNameParam = searchParams.get("projeto");

  const [selectedImage, setSelectedImage] = useState(null);
  const [imagens, setImagens] = useState([]);
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
          name: image.nomeArquivo,
          enviado: image.processada, // Use processada as enviado status
          enviadoPor: image.processada ? 'Sistema' : '',
          dataCaptura: image.dataCaptura,
          dataUpload: image.dataUpload,
          fachada: image.fachada?.nome || 'Sem fachada',
          edificio: image.fachada?.edificio?.nome || 'Sem edifício'
        }));
        
        setImagens(transformedImages);
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
            isNew: true // Mark as new upload
          },
        ]);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDeleteImage = (id) => {
    setImagens((prev) => prev.filter((img) => img.id !== id));
    // TODO: Implement API call to delete image from backend if not a new upload
    console.warn('Image deleted locally. Backend integration needed for persistence.');
  };

  const handleEnviarParaModelo = () => {
    // Get current user name - in a real app this would come from authentication context
    const nomeUsuario = localStorage.getItem('userName') || 'Usuário Atual';
    setImagens((prev) =>
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

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-100 p-8">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg">Carregando imagens...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-100 p-8">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <strong>Erro:</strong> {error}
        </div>
      </div>
    );
  }

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

      {imagens.length === 0 ? (
        <div className="text-center text-gray-500 py-8">
          Nenhuma imagem encontrada para este projeto.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {imagens.map((imagem) => (
            <div key={imagem.id} className="relative border rounded-lg p-4 bg-white shadow">
              {/* Status */}
              <div className="mb-2">
                {imagem.enviado ? (
                  <span className="text-green-700 bg-green-100 px-2 py-1 rounded text-sm font-medium">
                    Processada {imagem.enviadoPor && `por ${imagem.enviadoPor}`}
                  </span>
                ) : (
                  <span className="text-yellow-800 bg-yellow-100 px-2 py-1 rounded text-sm font-medium">
                    {imagem.isNew ? 'Nova imagem - Aguardando upload' : 'Aguardando processamento'}
                  </span>
                )}
              </div>

              {/* Additional info for existing images */}
              {!imagem.isNew && (
                <div className="mb-2 text-xs text-gray-600">
                  <div>Fachada: {imagem.fachada}</div>
                  <div>Edifício: {imagem.edificio}</div>
                </div>
              )}

              {/* Imagem */}
              <img
                src={imagem.src}
                alt={imagem.name}
                className="w-full h-48 object-contain rounded cursor-pointer"
                onClick={() => setSelectedImage(imagem)}
                onError={(e) => {
                  e.target.src = placeholder;
                }}
              />

              {/* Delete button */}
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

      {/* Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
          <div className="bg-white rounded p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-lg">
            <h2 className="text-xl font-semibold mb-4">{selectedImage.name}</h2>
            
            {/* Image details for existing images */}
            {!selectedImage.isNew && (
              <div className="mb-4 text-sm text-gray-600">
                <div><strong>Fachada:</strong> {selectedImage.fachada}</div>
                <div><strong>Edifício:</strong> {selectedImage.edificio}</div>
                {selectedImage.dataCaptura && (
                  <div><strong>Data de Captura:</strong> {new Date(selectedImage.dataCaptura).toLocaleDateString()}</div>
                )}
                <div><strong>Data de Upload:</strong> {new Date(selectedImage.dataUpload).toLocaleDateString()}</div>
              </div>
            )}

            {/* Image container with fixed height */}
            <div className="w-full h-80 mb-4 bg-gray-100 rounded flex items-center justify-center overflow-hidden">
              <img
                src={selectedImage.src}
                alt={selectedImage.name}
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
                  {selectedImage.isNew ? 'Fazer upload e enviar para o modelo' : 'Enviar para o modelo'}
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

export default UploadImagens;
