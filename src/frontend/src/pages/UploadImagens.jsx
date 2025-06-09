import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Trash2, Upload, AlertCircle, CheckCircle } from "lucide-react";
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

  // New states for building and facade selection
  const [edificios, setEdificios] = useState([]);
  const [fachadas, setFachadas] = useState([]);
  const [selectedEdificio, setSelectedEdificio] = useState('');
  const [selectedFachada, setSelectedFachada] = useState('');
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');

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

  // Fetch buildings when project is resolved
  useEffect(() => {
    const fetchEdificios = async () => {
      if (!resolvedProjectId) return;
      
      try {
        const projectName = projectNameParam || (await getProjectName(resolvedProjectId));
        if (!projectName) return;

        const response = await fetch(`http://localhost:8080/api/edificio/projeto-nome/${encodeURIComponent(projectName)}`);
        if (response.ok) {
          const data = await response.json();
          setEdificios(data || []);
        } else if (response.status === 204) {
          setEdificios([]);
        }
      } catch (err) {
        console.error('Erro ao buscar edifícios:', err);
        setEdificios([]);
      }
    };

    fetchEdificios();
  }, [resolvedProjectId, projectNameParam]);

  // Fetch facades when building is selected
  useEffect(() => {
    if (selectedEdificio) {
      const edificio = edificios.find(e => e.id.toString() === selectedEdificio);
      if (edificio && edificio.fachadas && edificio.fachadas.length > 0) {
        setFachadas(edificio.fachadas);
        setSelectedFachada(''); // Reset facade selection
      } else {
        setFachadas([]);
      }
    } else {
      setFachadas([]);
      setSelectedFachada('');
    }
  }, [selectedEdificio, edificios]);

  const getProjectName = async (projectId) => {
    try {
      const response = await fetch('http://localhost:8080/api/projetos');
      if (!response.ok) return null;
      const projects = await response.json();
      const project = projects.find(p => p.id.toString() === projectId.toString());
      return project ? project.nome : null;
    } catch (err) {
      console.error('Erro ao buscar nome do projeto:', err);
      return null;
    }
  };

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
        fachada: img.fachada?.descricao || 'Sem fachada', // Usar descricao em vez de nome
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
    const files = Array.from(event.target.files);
    if (!files.length || !resolvedProjectId) return;

    if (!selectedEdificio) {
      alert('Por favor, selecione um edifício antes de fazer o upload.');
      return;
    }

    if (!selectedFachada) {
      alert('Por favor, selecione uma fachada antes de fazer o upload.');
      return;
    }

    setUploadingFiles(true);
    setUploadProgress('Preparando upload...');

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append("files", file);
      });

      // Get the selected facade info
      const fachada = fachadas.find(f => f.id?.toString() === selectedFachada || f.descricao === selectedFachada);
      const direction = fachada ? fachada.descricao.toLowerCase() : 'general';

      setUploadProgress(`Fazendo upload de ${files.length} arquivo(s)...`);

      const response = await fetch(`http://localhost:8080/api/images/${resolvedProjectId}/upload/${selectedEdificio}/${direction}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Erro no upload: ${response.status} - ${errorText}`);
      }

      setUploadProgress('Upload concluído! Atualizando lista...');
      await fetchImages(resolvedProjectId);
      
      // Reset form
      event.target.value = '';
      setUploadProgress('');
      
      // Show success message
      setTimeout(() => {
        setUploadProgress('');
      }, 2000);

    } catch (err) {
      console.error("Erro no upload:", err);
      setError(`Erro no upload: ${err.message}`);
      setUploadProgress('');
    } finally {
      setUploadingFiles(false);
    }
  };

  const handleDeleteImage = async (id) => {
    if (!window.confirm('Tem certeza que deseja deletar esta imagem? Esta ação não pode ser desfeita.')) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8080/api/images/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Erro na resposta do servidor:', response.status, errorText);
        throw new Error(`Erro ${response.status}: ${errorText || 'Erro ao deletar imagem'}`);
      }

      // Remove a imagem da lista local apenas se a deleção foi bem-sucedida
      setImagens(prev => prev.filter(img => img.id !== id));
      
      // Se a imagem deletada estava selecionada, feche o modal
      if (selectedImage && selectedImage.id === id) {
        setSelectedImage(null);
      }

      console.log('Imagem deletada com sucesso');
    } catch (err) {
      console.error("Erro ao deletar imagem:", err);
      alert('Erro ao deletar imagem: ' + err.message);
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

          // Create fissura record in backend with correct format
          try {
            const fissuraData = {
              imagem: {
                id: resultado.id
              },
              tipo: resultado.label,
              coordenadas: resultado.coords ? JSON.stringify({
                x1: resultado.coords.x1,
                y1: resultado.coords.y1,
                x2: resultado.coords.x2,
                y2: resultado.coords.y2,
                width: resultado.coords.x2 - resultado.coords.x1,
                height: resultado.coords.y2 - resultado.coords.y1
              }) : null,
              gravidade: resultado.severity || 'Baixa',
              confianca: resultado.confidence,
              dataDeteccao: new Date().toISOString(),
              aprovado: false,
              aprovadoPor: null
            };

            console.log('Enviando dados da fissura:', fissuraData);

            const fissuraResponse = await fetch('http://localhost:8080/api/fissura', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(fissuraData)
            });

            if (!fissuraResponse.ok) {
              const errorText = await fissuraResponse.text();
              console.error('Erro ao salvar fissura:', errorText);
              throw new Error(`Erro ${fissuraResponse.status}: ${errorText}`);
            }

            console.log('Fissura salva com sucesso');
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

  if (isLoading) return (
    <div className="min-h-screen bg-slate-100 p-8">
      <div className="flex justify-center items-center h-64">
        <div className="text-lg">Carregando dados do projeto...</div>
      </div>
    </div>
  );
  
  if (error) return (
    <div className="min-h-screen bg-slate-100 p-8">
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
        <div className="flex items-center">
          <AlertCircle className="mr-2" size={20} />
          <span>Erro: {error}</span>
        </div>
      </div>
      <button 
        onClick={() => window.location.reload()} 
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Tentar Novamente
      </button>
    </div>
  );

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

      {/* Upload progress indicator */}
      {uploadingFiles && (
        <div className="fixed top-4 left-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded shadow-lg z-50">
          <div className="flex items-center">
            <Upload className="mr-2" size={20} />
            <span>{uploadProgress}</span>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-dark-blue">Upload de Imagens</h1>
        
        {projectNameParam && (
          <p className="text-gray-600 mb-6">Projeto: <span className="font-semibold">{projectNameParam}</span></p>
        )}

        {/* Building and Facade Selection */}
        <div className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Selecionar Localização</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="edificio" className="block text-sm font-medium text-gray-700 mb-2">
                Edifício *
              </label>
              <select
                id="edificio"
                value={selectedEdificio}
                onChange={(e) => setSelectedEdificio(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={uploadingFiles}
              >
                <option value="">Selecione um edifício</option>
                {edificios.map((edificio) => (
                  <option key={edificio.id} value={edificio.id}>
                    {edificio.nome} - {edificio.localizacao}
                  </option>
                ))}
              </select>
              {edificios.length === 0 && (
                <p className="text-sm text-amber-600 mt-1">
                  Nenhum edifício encontrado. Cadastre edifícios no projeto primeiro.
                </p>
              )}
            </div>

            <div>
              <label htmlFor="fachada" className="block text-sm font-medium text-gray-700 mb-2">
                Fachada *
              </label>
              <select
                id="fachada"
                value={selectedFachada}
                onChange={(e) => setSelectedFachada(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={!selectedEdificio || uploadingFiles}
              >
                <option value="">Selecione uma fachada</option>
                {fachadas.map((fachada, index) => (
                  <option key={fachada.id || index} value={fachada.id || fachada.descricao}>
                    {fachada.descricao} {fachada.area && `(${fachada.area} m²)`}
                  </option>
                ))}
              </select>
              {selectedEdificio && fachadas.length === 0 && (
                <p className="text-sm text-amber-600 mt-1">
                  Nenhuma fachada encontrada para este edifício.
                </p>
              )}
            </div>
          </div>

          <div className="text-sm text-gray-600">
            <p>* Campos obrigatórios para realizar o upload das imagens.</p>
          </div>
        </div>

        {/* Upload Area */}
        <div className="bg-white p-8 rounded-lg shadow-md mb-8">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
            {uploadingFiles ? (
              <div className="py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">{uploadProgress}</p>
              </div>
            ) : (
              <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <label className="cursor-pointer">
                  <span className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors inline-block">
                    Selecionar Imagens
                  </span>
                  <input 
                    type="file" 
                    accept="image/*" 
                    multiple 
                    className="hidden" 
                    onChange={handleImageUpload}
                    disabled={!selectedEdificio || !selectedFachada}
                  />
                </label>
                <p className="text-gray-500 mt-2">
                  Selecione múltiplas imagens (JPG, PNG, etc.)
                </p>
                {(!selectedEdificio || !selectedFachada) && (
                  <p className="text-amber-600 text-sm mt-2">
                    Selecione um edifício e fachada antes de fazer o upload
                  </p>
                )}
              </>
            )}
          </div>
        </div>

        <h2 className="text-2xl font-bold mb-6 text-dark-blue">Imagens Carregadas</h2>

        {imagens.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg shadow-md">
            <div className="text-gray-500 text-lg">Nenhuma imagem encontrada.</div>
            <p className="text-gray-400 mt-2">Faça o upload de imagens para começar.</p>
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: '1.5rem',
            gridAutoRows: 'max-content'
          }}>
            {imagens.map(imagem => (
              <div key={imagem.id} className="relative bg-white rounded-lg shadow-md overflow-hidden" style={{ breakInside: 'avoid' }}>
                {/* Status Badge */}
                <div className="absolute top-3 left-3 z-10">
                  {imagem.enviado ? (
                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium flex items-center">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      Processada {imagem.enviadoPor && `por ${imagem.enviadoPor}`}
                    </span>
                  ) : (
                    <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">
                      Aguardando processamento
                    </span>
                  )}
                </div>

                {/* Delete Button */}
                <button
                  onClick={() => handleDeleteImage(imagem.id)}
                  className="absolute top-3 right-3 z-10 bg-red-100 hover:bg-red-200 text-red-600 p-2 rounded-full transition-colors"
                  title="Deletar imagem"
                >
                  <Trash2 size={16} />
                </button>

                {/* Image Container with dynamic aspect ratio */}
                <div className="relative bg-gray-100 overflow-hidden">
                  <img
                    src={imagem.src}
                    alt={imagem.name}
                    className="w-full h-auto cursor-pointer hover:scale-105 transition-transform"
                    onClick={() => setSelectedImage(imagem)}
                    onError={(e) => { e.target.src = placeholder }}
                    style={{
                      maxHeight: '300px',
                      objectFit: 'cover',
                      aspectRatio: 'auto'
                    }}
                  />
                </div>

                {/* Loading overlay for processing image */}
                {processingImage === imagem.id && (
                  <div className="absolute inset-0 bg-blue-100 bg-opacity-90 flex items-center justify-center">
                    <div className="text-blue-700 font-medium text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-700 mx-auto mb-2"></div>
                      Processando...
                    </div>
                  </div>
                )}

                {/* Image Info */}
                <div className="p-4">
                  <h3 className="font-medium text-gray-900 mb-2 truncate" title={imagem.name}>
                    {imagem.name}
                  </h3>
                  
                  <div className="space-y-1 text-xs text-gray-600">
                    <div className="flex justify-between">
                      <span>Edifício:</span>
                      <span className="font-medium">{imagem.edificio}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Fachada:</span>
                      <span className="font-medium">{imagem.fachada}</span>
                    </div>
                    {imagem.tipo && (
                      <div className="flex justify-between">
                        <span>Tipo:</span>
                        <span className="font-medium">{imagem.tipo}</span>
                      </div>
                    )}
                    {imagem.confianca && (
                      <div className="flex justify-between">
                        <span>Confiança:</span>
                        <span className="font-medium">{(imagem.confianca * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Modal */}
        {selectedImage && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <h2 className="text-2xl font-semibold mb-4">{selectedImage.name}</h2>
                
                {/* Image details */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-800 mb-3">Informações da Imagem</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Edifício:</span>
                        <span className="font-medium">{selectedImage.edificio}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Fachada:</span>
                        <span className="font-medium">{selectedImage.fachada}</span>
                      </div>
                      {selectedImage.dataCaptura && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Data de Captura:</span>
                          <span className="font-medium">{new Date(selectedImage.dataCaptura).toLocaleDateString()}</span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-gray-600">Data de Upload:</span>
                        <span className="font-medium">{new Date(selectedImage.dataUpload).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>

                  {(selectedImage.tipo || selectedImage.confianca || selectedImage.coordenadas) && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h3 className="font-semibold text-gray-800 mb-3">Análise</h3>
                      <div className="space-y-2 text-sm">
                        {selectedImage.tipo && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Tipo:</span>
                            <span className="font-medium">{selectedImage.tipo}</span>
                          </div>
                        )}
                        {selectedImage.confianca && (
                          <div className="flex justify-between">
                            <span className="text-gray-600">Confiança:</span>
                            <span className="font-medium">{(selectedImage.confianca * 100).toFixed(1)}%</span>
                          </div>
                        )}
                        {selectedImage.coordenadas && (
                          <div>
                            <span className="text-gray-600">Coordenadas:</span>
                            <div className="text-xs font-mono mt-1 bg-white p-2 rounded">
                              x: {selectedImage.coordenadas.x}, y: {selectedImage.coordenadas.y}<br/>
                              w: {selectedImage.coordenadas.width}, h: {selectedImage.coordenadas.height}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Image container */}
                <div className="bg-gray-100 rounded-lg p-4 mb-6">
                  <img
                    src={selectedImage.src}
                    alt={selectedImage.name}
                    className="max-w-full max-h-96 mx-auto object-contain rounded"
                    onError={(e) => { e.target.src = placeholder }}
                  />
                </div>
                
                {/* Actions */}
                <div className="flex justify-between items-center">
                  {!selectedImage.enviado ? (
                    <button
                      onClick={handleEnviarParaModelo}
                      disabled={processingImage !== null}
                      className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      {processingImage === selectedImage.id ? 'Processando...' : 'Enviar para Análise'}
                    </button>
                  ) : (
                    <div className="flex items-center text-green-600 font-medium">
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Já processada {selectedImage.enviadoPor && `por ${selectedImage.enviadoPor}`}
                    </div>
                  )}
                  <button
                    onClick={() => setSelectedImage(null)}
                    className="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600 transition-colors"
                    disabled={processingImage === selectedImage.id}
                  >
                    Fechar
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadImagens;
