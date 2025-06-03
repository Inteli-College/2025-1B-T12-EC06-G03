import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import placeholder from '../assets/placeholder-icon.svg';
import { Trash2 } from 'lucide-react';
import imageInferenceService, { getCurrentUser } from '../utils/imageInference';

const DroneImages = () => {
  const [searchParams] = useSearchParams();
  const projectIdParam = searchParams.get("id");
  const projectNameParam = searchParams.get("projeto");

  const [selectedImage, setSelectedImage] = useState(null);
  const [images, setImages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resolvedProjectId, setResolvedProjectId] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);
  const [processingImage, setProcessingImage] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');

  // Supabase base URL for images
  const SUPABASE_BASE_URL = "https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/";

  // Get current user on component mount
  useEffect(() => {
    const fetchCurrentUser = async () => {
      const user = await getCurrentUser();
      setCurrentUser(user);
    };
    fetchCurrentUser();
  }, []);

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
          enviadoPor: image.processadaPor || (image.processada ? 'Sistema' : ''),
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

  const handleEnviarParaModelo = async () => {
    if (!selectedImage || !currentUser) {
      alert('Erro: Usuário não identificado ou imagem não selecionada');
      return;
    }

    try {
      setProcessingImage(selectedImage.id);
      setProcessingStatus('Conectando ao servidor...');

      // Connect to WebSocket
      await imageInferenceService.connect();
      
      // Update UI to show processing
      const userName = currentUser.nome || 'Usuário Atual';
      setImages((prev) =>
        prev.map((img) =>
          img.id === selectedImage.id
            ? { ...img, enviado: true, enviadoPor: userName }
            : img
        )
      );

      // Persist status in backend
      await fetch(`http://localhost:8080/api/images/${selectedImage.id}/processada`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          processada: true,
          processadaPor: userName
        })
      });

      // Send image for processing
      await imageInferenceService.inferImages([selectedImage.id], {
        onStatus: (data) => {
          setProcessingStatus(data.message);
        },
        onResults: (results) => {
          console.log('Resultados do processamento:', results);
          // Here you could update images with classification results if needed
          // For now, we just keep the processing status
        },
        onError: (error) => {
          console.error('Erro no processamento:', error);
          setProcessingStatus('Erro no processamento');
          // Revert the UI change on error
          setImages((prev) =>
            prev.map((img) =>
              img.id === selectedImage.id
                ? { ...img, enviado: false, enviadoPor: '' }
                : img
            )
          );
          alert('Erro ao processar imagem: ' + error.error);
        },
        onComplete: (data) => {
          setProcessingStatus('Processamento concluído!');
          setTimeout(() => {
            setProcessingStatus('');
            setProcessingImage(null);
          }, 2000);
        }
      });

      setSelectedImage(null);
      
    } catch (error) {
      console.error('Erro ao enviar para modelo:', error);
      setProcessingStatus('');
      setProcessingImage(null);
      
      // Revert the UI change on error
      setImages((prev) =>
        prev.map((img) =>
          img.id === selectedImage.id
            ? { ...img, enviado: false, enviadoPor: '' }
            : img
        )
      );
      
      alert('Erro ao conectar com o servidor de processamento: ' + error.message);
    }
  };

  const handleDeleteImage = async (id) => {
    try {
      const response = await fetch(`http://localhost:8080/api/images/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Erro ao deletar');

      setImages((prev) => prev.filter((img) => img.id !== id));
    } catch (err) {
      console.error("Erro ao deletar imagem:", err);
      alert('Erro ao deletar imagem: ' + err.message);
    }
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
      {/* Processing status indicator */}
      {processingImage && (
        <div className="fixed top-4 right-4 bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded shadow-lg z-50">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 mr-2"></div>
            <span>{processingStatus}</span>
          </div>
        </div>
      )}

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

                {/* Loading overlay for processing image */}
                {processingImage === imagem.id && (
                  <div className="absolute inset-0 bg-blue-100 bg-opacity-75 flex items-center justify-center rounded-lg">
                    <div className="text-blue-700 font-medium">Processando...</div>
                  </div>
                )}

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
                  disabled={processingImage !== null}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {processingImage === selectedImage.id ? 'Processando...' : 'Enviar para o modelo'}
                </button>
              ) : (
                <p className="text-green-600 font-medium">
                  Já processada {selectedImage.enviadoPor && `por ${selectedImage.enviadoPor}`}
                </p>
              )}
              <button
                onClick={() => setSelectedImage(null)}
                className="text-sm text-gray-500 hover:underline ml-4"
                disabled={processingImage === selectedImage.id}
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