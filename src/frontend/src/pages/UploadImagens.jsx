import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Trash2 } from "lucide-react";
import placeholder from "../assets/placeholder-icon.svg";
import imageInferenceService, { getCurrentUser } from '../utils/imageInference';

const SUPABASE_BASE_URL = "https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/";

const UploadImagens = () => {
  const [searchParams] = useSearchParams();
  const projectIdParam = searchParams.get("id");
  const projectNameParam = searchParams.get("projeto");

  const [selectedImage, setSelectedImage] = useState(null);
  const [imagens, setImagens] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resolvedProjectId, setResolvedProjectId] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);
  const [processingImage, setProcessingImage] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');

  // Get current user on component mount
  useEffect(() => {
    const fetchCurrentUser = async () => {
      const user = await getCurrentUser();
      setCurrentUser(user);
    };
    fetchCurrentUser();
  }, []);

  const resolveProjectId = async (projectName) => {
    try {
      const response = await fetch('http://localhost:8080/api/projetos');
      if (!response.ok) throw new Error('Failed to fetch projects');
      const projects = await response.json();
      const project = projects.find(p => p.nome.toLowerCase() === projectName.toLowerCase());
      return project ? project.id : null;
    } catch (err) {
      console.error('Erro ao buscar projetos:', err);
      return null;
    }
  };

  useEffect(() => {
    const initializeProjectId = async () => {
      let projectId = projectIdParam || await resolveProjectId(projectNameParam);
      if (!projectId) {
        setError('ID ou nome do projeto não encontrado');
        setIsLoading(false);
        return;
      }
      setResolvedProjectId(projectId);
    };
    initializeProjectId();
  }, [projectIdParam, projectNameParam]);

  const fetchImages = async (projectId) => {
    try {
      const response = await fetch(`http://localhost:8080/api/images/${projectId}`);
      if (!response.ok) throw new Error(`Erro HTTP: ${response.status}`);
      const result = await response.json();
      const transformed = result.map(img => ({
        id: img.id,
        src: SUPABASE_BASE_URL + img.caminhoArquivo,
        name: img.nomeArquivo,
        enviado: img.processada,
        enviadoPor: img.processadaPor || (img.processada ? 'Sistema' : ''),
        dataCaptura: img.dataCaptura,
        dataUpload: img.dataUpload,
        fachada: img.fachada?.nome || 'Sem fachada',
        edificio: img.fachada?.edificio?.nome || 'Sem edifício',
        tipo: img.fissura?.tipo || null,
        coordenadas: img.fissura?.coordenadas || null,
        confianca: img.fissura?.confianca || null,
        gravidade: img.fissura?.gravidade || null
      }));
      setImagens(transformed);
      setError(null);
    } catch (err) {
      setError(err.message || 'Erro ao buscar imagens');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!resolvedProjectId) return;
    fetchImages(resolvedProjectId);
  }, [resolvedProjectId]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !resolvedProjectId) return;

    const formData = new FormData();
    formData.append("files", file);
    const edificioId = 1;
    const direction = "west";

    try {
      const response = await fetch(`http://localhost:8080/api/images/${resolvedProjectId}/upload/${edificioId}/${direction}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error('Erro no upload');

      await fetchImages(resolvedProjectId);
    } catch (err) {
      console.error("Erro no upload:", err);
    }
  };

  const handleDeleteImage = async (id) => {
    try {
      const response = await fetch(`http://localhost:8080/api/images/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Erro ao deletar');

      setImagens(prev => prev.filter(img => img.id !== id));
    } catch (err) {
      console.error("Erro ao deletar imagem:", err);
    }
  };

  const handleEnviarParaModelo = async () => {
    if (!selectedImage || !selectedImage.id || !currentUser) {
      console.error("Imagem selecionada inválida ou usuário não identificado");
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
      setImagens((prev) =>
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
        onResults: async (results) => {
          console.log('Resultados do processamento:', results);
          const resultado = results[0];

          // Create fissura record in backend
          try {
            await fetch('http://localhost:8080/api/fissuras', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                imagem_id: resultado.id,
                tipo: resultado.label,
                coordenadas: resultado.coords,
                gravidade: resultado.severity || 'moderada',
                data_deteccao: new Date().toISOString(),
                confianca: resultado.confidence
              })
            });
          } catch (fissuraError) {
            console.error('Erro ao salvar fissura:', fissuraError);
          }
        },
        onError: (error) => {
          console.error('Erro no processamento:', error);
          setProcessingStatus('Erro no processamento');
          // Revert the UI change on error
          setImagens((prev) =>
            prev.map((img) =>
              img.id === selectedImage.id
                ? { ...img, enviado: false, enviadoPor: '' }
                : img
            )
          );
          alert('Erro ao processar imagem: ' + error.error);
        },
        onComplete: async (data) => {
          setProcessingStatus('Processamento concluído! Atualizando dados...');
          
          // Refresh images to get updated data
          await fetchImages(resolvedProjectId);
          
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
      setImagens((prev) =>
        prev.map((img) =>
          img.id === selectedImage.id
            ? { ...img, enviado: false, enviadoPor: '' }
            : img
        )
      );
      
      alert('Erro ao conectar com o servidor de processamento: ' + error.message);
    }
  };

  if (isLoading) return <div className="p-8">Carregando imagens...</div>;
  if (error) return <div className="p-8 text-red-600">Erro: {error}</div>;

  return (
    <div className="min-h-screen bg-slate-100 p-8">
      {/* Processing status indicator */}
      {processingImage && (
        <div className="fixed top-4 right-4 bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded shadow-lg z-50">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-700 mr-2"></div>
            <span>{processingStatus}</span>
          </div>
        </div>
      )}

      <h1 className="text-3xl font-bold mb-6 text-dark-blue">Upload de Imagem</h1>

      <div className="bg-gray-light h-72 flex items-center justify-center rounded-md mb-10">
        <label className="bg-dark-blue text-gray-light px-6 py-2 rounded-xl shadow-md cursor-pointer">
          Carregar Imagem
          <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
        </label>
      </div>

      <h2 className="text-3xl font-bold mb-6 text-dark-blue">Imagens Carregadas</h2>

      {imagens.length === 0 ? (
        <div className="text-center text-gray-500 py-8">Nenhuma imagem encontrada.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {imagens.map(imagem => (
            <div key={imagem.id} className="relative border rounded-lg p-4 bg-white shadow">
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

              <div className="mb-2 text-xs text-gray-600">
                <div>Fachada: {imagem.fachada}</div>
                <div>Edifício: {imagem.edificio}</div>
                {imagem.tipo && <div>Tipo: {imagem.tipo}</div>}
                {imagem.confianca && <div>Confiança: {(imagem.confianca * 100).toFixed(1)}%</div>}
                {imagem.coordenadas && (
                  <div>Coordenadas: x={imagem.coordenadas.x}, y={imagem.coordenadas.y}, w={imagem.coordenadas.width}, h={imagem.coordenadas.height}</div>
                )}
              </div>

              <img
                src={imagem.src}
                alt={imagem.name}
                className="w-full h-48 object-contain rounded cursor-pointer"
                onClick={() => setSelectedImage(imagem)}
                onError={(e) => { e.target.src = placeholder }}
              />

              {/* Loading overlay for processing image */}
              {processingImage === imagem.id && (
                <div className="absolute inset-0 bg-blue-100 bg-opacity-75 flex items-center justify-center rounded-lg">
                  <div className="text-blue-700 font-medium">Processando...</div>
                </div>
              )}

              <button
                onClick={() => handleDeleteImage(imagem.id)}
                className="absolute top-2 right-2 text-red-500 hover:text-red-700"
              >
                <Trash2 size={20} />
              </button>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
          <div className="bg-white rounded p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-lg">
            <h2 className="text-xl font-semibold mb-4">{selectedImage.name}</h2>

            <div className="w-full h-80 mb-4 bg-gray-100 rounded flex items-center justify-center overflow-hidden">
              <img
                src={selectedImage.src}
                alt={selectedImage.name}
                className="max-w-full max-h-full object-contain"
                onError={(e) => { e.target.src = placeholder }}
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

export default UploadImagens;
