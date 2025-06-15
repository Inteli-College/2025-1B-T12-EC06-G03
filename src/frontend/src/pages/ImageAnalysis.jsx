import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import ImageCard from "../components/ProjectImageCard";
import placeholder from "../assets/placeholder-icon.svg";

export default function ImageAnalysis() {
  const [searchParams] = useSearchParams();
  const projetoAtivo = searchParams.get("projeto");

  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Função para traduzir tipos de fissura
  const traduzirTipoFissura = (tipo) => {
    const traducoes = {
      'retraction': 'Retração',
      'thermal': 'Térmica'
    };
    return traducoes[tipo?.toLowerCase()] || tipo;
  };

  useEffect(() => {
    const fetchFissuras = async () => {
      if (!projetoAtivo) {
        setLoading(false);
        return;
      }

      try {
        // Primeiro, buscar o ID do projeto pelo nome
        const projetosResponse = await fetch('http://localhost:8080/api/projetos');
        if (!projetosResponse.ok) {
          throw new Error('Erro ao buscar projetos');
        }
        const projetos = await projetosResponse.json();
        const projeto = projetos.find(p => p.nome.toLowerCase() === projetoAtivo.toLowerCase());
        
        if (!projeto) {
          throw new Error(`Projeto "${projetoAtivo}" não encontrado`);
        }

        // Buscar fissuras detalhadas do projeto
        const fissurasResponse = await fetch(`http://localhost:8080/api/fissura/detalhes/projeto/${projeto.id}`);
        if (!fissurasResponse.ok) {
          throw new Error('Erro ao buscar fissuras do projeto');
        }
        const fissuras = await fissurasResponse.json();

        // Formatar dados para exibição
        const formatadas = fissuras.map(f => {
          let bbox = null;
          try {
            bbox = f.coordenadas ? JSON.parse(f.coordenadas) : null;
          } catch (e) {
            console.warn('Erro ao parsear coordenadas:', f.coordenadas);
          }

          return {
            id: f.id,
            fissuraId: f.id,
            caminho: `https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/${f.nomeImagem}`,
            label: traduzirTipoFissura(f.tipo), // Aplicar tradução aqui
            labelOriginal: f.tipo, // Manter original para referência
            bbox: bbox,
            gravidade: f.gravidade,
            dataDeteccao: f.dataDeteccao,
            confidence: f.confianca,
            aprovado: f.aprovado || false,
            aprovadoPor: f.aprovadoPor || null,
          };
        });

        setImages(formatadas);
        setError(null);
      } catch (err) {
        console.error("Erro ao carregar fissuras:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchFissuras();
  }, [projetoAtivo]);

  const handleViewImage = (image) => {
    setSelectedImage(image);
  };

  const handleAprovar = async () => {
    try {
      // Buscar informações do usuário logado
      const token = localStorage.getItem('token');
      const userResponse = await fetch('http://localhost:8080/auth/@me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!userResponse.ok) {
        throw new Error('Erro ao obter dados do usuário');
      }
      
      const userData = await userResponse.json();
      const nomeUsuario = userData.nome || 'Usuário';

      // Atualizar estado local
      setImages((prev) =>
        prev.map((img) =>
          img.id === selectedImage.id
            ? { ...img, aprovado: true, aprovadoPor: nomeUsuario }
            : img
        )
      );

      // Enviar aprovação para o backend
      await fetch(`http://localhost:8080/api/fissura/${selectedImage.fissuraId}/aprovar`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          aprovado: true, 
          aprovadoPor: nomeUsuario 
        })
      });

      setSelectedImage(null);
    } catch (err) {
      console.error('Erro ao aprovar:', err);
      alert('Erro ao aprovar fissura: ' + err.message);
    }
  };

  const formatarData = (dataString) => {
    if (!dataString) return 'N/A';
    try {
      const data = new Date(dataString);
      return data.toLocaleDateString('pt-BR') + ' ' + data.toLocaleTimeString('pt-BR');
    } catch (e) {
      return dataString;
    }
  };

  const formatarCoordenadas = (coordenadas) => {
    if (!coordenadas) return 'N/A';
    try {
      if (typeof coordenadas === 'string') {
        coordenadas = JSON.parse(coordenadas);
      }
      // Novo formato com x1, y1, x2, y2
      if (coordenadas.x1 !== undefined && coordenadas.y1 !== undefined && 
          coordenadas.x2 !== undefined && coordenadas.y2 !== undefined) {
        return `x1: ${coordenadas.x1}, y1: ${coordenadas.y1}, x2: ${coordenadas.x2}, y2: ${coordenadas.y2}, largura: ${coordenadas.width || coordenadas.x2 - coordenadas.x1}, altura: ${coordenadas.height || coordenadas.y2 - coordenadas.y1}`;
      }
      // Formato antigo como fallback
      return `x: ${coordenadas.x || 'N/A'}, y: ${coordenadas.y || 'N/A'}, largura: ${coordenadas.width || 'N/A'}, altura: ${coordenadas.height || 'N/A'}`;
    } catch (e) {
      return 'Formato inválido';
    }
  };

  const getGravidadeColor = (gravidade) => {
    switch (gravidade?.toLowerCase()) {
      case 'alta':
        return 'bg-red-100 text-red-800';
      case 'média':
      case 'media':
        return 'bg-yellow-100 text-yellow-800';
      case 'baixa':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getTipoColor = (tipo) => {
    switch (tipo?.toLowerCase()) {
      case 'térmica':
      case 'termica':
      case 'thermal':
        return 'bg-orange-100 text-orange-800';
      case 'retração':
      case 'retracao':
      case 'retraction':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-purple-100 text-purple-800';
    }
  };

  if (loading) {
    return (
      <main className="container mx-auto p-6">
        <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>
        <div className="flex justify-center items-center h-64">
          <div className="text-lg">Carregando imagens analisadas...</div>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="container mx-auto p-6">
        <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-red-600">
            Erro: {error}
          </div>
        </div>
      </main>
    );
  }

  if (!projetoAtivo) {
    return (
      <main className="container mx-auto p-6">
        <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-yellow-600">
            Por favor, selecione um projeto na URL (ex: ?projeto=NomeDoProjeto)
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="container mx-auto p-6">
      <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>
      
      {images.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-lg text-gray-600">
            Nenhuma fissura foi encontrada para o projeto "{projetoAtivo}".
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {images.map((image) => (
            <div key={`${image.id}-${image.fissuraId}`} className="relative border rounded-lg p-4 bg-white shadow-lg">
              {/* Status de Aprovação */}
              <div className="mb-3">
                {image.aprovado ? (
                  <span className="text-green-700 bg-green-100 px-3 py-1 rounded-full text-sm font-medium">
                    ✓ Aprovado {image.aprovadoPor && `por ${image.aprovadoPor}`}
                  </span>
                ) : (
                  <span className="text-yellow-800 bg-yellow-100 px-3 py-1 rounded-full text-sm font-medium">
                    ⏳ Aguardando Aprovação
                  </span>
                )}
              </div>

              {/* Detalhes da Classificação */}
              <div className="mb-3 space-y-2">
                <div className="flex flex-wrap gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getTipoColor(image.label)}`}>
                    Tipo: {image.label || 'Não classificado'}
                  </span>
                  {image.gravidade && (
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getGravidadeColor(image.gravidade)}`}>
                      {image.gravidade}
                    </span>
                  )}
                </div>
                
                <div className="text-xs text-gray-600 space-y-1">
                  <div>
                    <strong>Confiança:</strong> {image.confidence ? `${(image.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                  <div>
                    <strong>Detectado em:</strong> {formatarData(image.dataDeteccao)}
                  </div>
                </div>
              </div>

              {/* Card da Imagem */}
              <ImageCard
                id={image.id}
                type={image.label}
                imageUrl={image.caminho}
                onView={() => handleViewImage(image)}
              />
            </div>
          ))}
        </div>
      )}

      {/* Modal de Detalhes */}
      {selectedImage && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50 p-4">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-4xl w-full max-h-[95vh] overflow-y-auto">
            <h2 className="text-xl font-semibold mb-4">
              Detalhes da Fissura #{selectedImage.fissuraId}
            </h2>
            
            <div className="mb-4 flex justify-center">
              <img
                src={selectedImage.caminho || placeholder}
                alt={`Fissura ${selectedImage.fissuraId}`}
                className="rounded object-contain border"
                style={{ width: '600px', height: '400px' }}
              />
            </div>

            <div className="space-y-4">
              {/* Classificação */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">Classificação</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <span className="text-sm font-medium text-gray-600">Tipo:</span>
                    <div className={`inline-block ml-2 px-2 py-1 rounded text-sm ${getTipoColor(selectedImage.label)}`}>
                      {selectedImage.label || 'Não classificado'}
                    </div>
                  </div>
                  {selectedImage.gravidade && (
                    <div>
                      <span className="text-sm font-medium text-gray-600">Gravidade:</span>
                      <div className={`inline-block ml-2 px-2 py-1 rounded text-sm ${getGravidadeColor(selectedImage.gravidade)}`}>
                        {selectedImage.gravidade}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Dados Técnicos */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">Dados Técnicos</h3>
                <div className="space-y-2 text-sm">
                  <div>
                    <strong>Confiança da Detecção:</strong> 
                    <span className="ml-2">
                      {selectedImage.confidence ? `${(selectedImage.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div>
                    <strong>Coordenadas:</strong>
                    <div className="ml-2 text-gray-600 font-mono text-xs mt-1">
                      {formatarCoordenadas(selectedImage.bbox)}
                    </div>
                  </div>
                  <div>
                    <strong>Data de Detecção:</strong>
                    <span className="ml-2">{formatarData(selectedImage.dataDeteccao)}</span>
                  </div>
                </div>
              </div>

              {/* Status e Ações */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">Status</h3>
                {!selectedImage.aprovado ? (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-600">Esta fissura ainda não foi aprovada por um especialista.</p>
                    <button
                      onClick={handleAprovar}
                      className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                    >
                      Aprovar Classificação
                    </button>
                  </div>
                ) : (
                  <p className="text-green-600 font-medium">
                    ✓ Classificação aprovada por {selectedImage.aprovadoPor}
                  </p>
                )}
              </div>
            </div>

            <div className="mt-6 text-right">
              <button
                onClick={() => setSelectedImage(null)}
                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
              >
                Fechar
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
