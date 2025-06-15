import { useState } from 'react';
import placeholderIcon from '../assets/placeholder-icon.svg';

const ImagensCarregadas = (props) => {
  console.log(props);

  const [isModalOpen, setIsModalOpen] = useState(false);

  // Função para traduzir tipos de fissura
  const traduzirTipoFissura = (tipo) => {
    const traducoes = {
      'retraction': 'Retração',
      'thermal': 'Térmica'
    };
    return traducoes[tipo?.toLowerCase()] || tipo;
  };

  // Tenta converter coordenadas JSON para objeto
  let coords = null;
  try {
    if (props.coordenadas) coords = typeof props.coordenadas === "string" ? JSON.parse(props.coordenadas) : props.coordenadas;
  } catch (e) {
    coords = null;
  }

  const handleAprovar = async () => {
    if (props.onAprovar) {
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

        // Chamar a função de aprovação passada como prop
        await props.onAprovar(props.id, nomeUsuario);
      } catch (error) {
        console.error('Erro ao aprovar fissura:', error);
        alert('Erro ao aprovar fissura: ' + error.message);
      }
    }
    setIsModalOpen(false);
  };

  // Traduzir o tipo para exibição
  const tipoTraduzido = traduzirTipoFissura(props.tipo);

  return (
    <div>
      {/* Card pequeno: tipo apenas como texto no bloco azul */}
      <div className="bg-blue-dark rounded-md shadow-md p-2">
        <div
          className="relative bg-gray-light rounded-md overflow-hidden border border-gray-300"
          style={{ paddingTop: '75%' }} 
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src={props.imgSrc || placeholderIcon}
              alt={props.name || 'Imagem Placeholder'}
              className="w-full h-full object-contain cursor-pointer rounded" 
              onClick={() => setIsModalOpen(true)} 
            />
          </div>
        </div>
        {/* Tipo como texto simples - traduzido */}
        <div className="mt-2 text-white text-sm text-center font-semibold">
          {tipoTraduzido}
        </div>
        
        {/* Status de aprovação */}
        <div className="mt-1 text-center">
          {props.aprovado ? (
            <span className="text-green-300 text-xs">
              ✓ Aprovado {props.aprovadoPor && `por ${props.aprovadoPor}`}
            </span>
          ) : (
            <span className="text-yellow-300 text-xs">
              ⏳ Aguardando aprovação
            </span>
          )}
        </div>
      </div>

      {/* Modal com todas as informações */}
      {isModalOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setIsModalOpen(false)} 
        >
          <div className="relative bg-white rounded-md p-6 max-w-4xl w-full max-h-[95vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
            <div className="flex justify-center mb-4">
              <img
                src={props.imgSrc || placeholderIcon}
                alt={props.name || 'Imagem Ampliada'}
                className="rounded-md object-contain"
                style={{ width: '500px', height: '350px' }}
              />
            </div>
            <button
              className="absolute top-2 right-2 bg-white text-black rounded-full p-2"
              onClick={() => setIsModalOpen(false)}
            >
              ✕
            </button>
            <div className="mt-4 text-[#010131] text-sm">
              <div><strong>Nome:</strong> {props.name}</div>
              <div><strong>Tipo:</strong> {tipoTraduzido}</div>
              {props.gravidade && <div><strong>Gravidade:</strong> {props.gravidade}</div>}
              {props.confianca !== undefined && props.confianca !== null && (
                <div><strong>Acurácia:</strong> {(props.confianca * 100).toFixed(1)}%</div>
              )}
              {coords && (
                <div className="mt-2">
                  <strong>Coordenadas:</strong>
                  <div className="font-mono text-xs">
                    {coords.x1 !== undefined && coords.y1 !== undefined && coords.x2 !== undefined && coords.y2 !== undefined ? (
                      <>x1: {coords.x1}, y1: {coords.y1}, x2: {coords.x2}, y2: {coords.y2}, largura: {coords.width || coords.x2 - coords.x1}, altura: {coords.height || coords.y2 - coords.y1}</>
                    ) : (
                      <>x: {coords.x}, y: {coords.y}, largura: {coords.width || coords.w}, altura: {coords.height || coords.h}</>
                    )}
                  </div>
                </div>
              )}
              
              {/* Status e botão de aprovação */}
              <div className="mt-4 pt-4 border-t">
                {props.aprovado ? (
                  <div className="text-green-600 font-medium">
                    ✓ Classificação aprovada {props.aprovadoPor && `por ${props.aprovadoPor}`}
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="text-yellow-600">
                      ⏳ Esta fissura ainda não foi aprovada por um especialista
                    </div>
                    {props.onAprovar && (
                      <button
                        onClick={handleAprovar}
                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                      >
                        Aprovar Classificação
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImagensCarregadas;
