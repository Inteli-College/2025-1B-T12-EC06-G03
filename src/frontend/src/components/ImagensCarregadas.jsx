import { useState } from 'react';
import placeholderIcon from '../assets/placeholder-icon.svg';

const ImagensCarregadas = (props) => {
  console.log(props);

  const [isModalOpen, setIsModalOpen] = useState(false);

  // Tenta converter coordenadas JSON para objeto
  let coords = null;
  try {
    if (props.coordenadas) coords = typeof props.coordenadas === "string" ? JSON.parse(props.coordenadas) : props.coordenadas;
  } catch (e) {
    coords = null;
  }

  const handleAprovar = () => {
    if (props.onAprovar) {
      props.onAprovar(props.id);
    }
    setIsModalOpen(false);
  };

  return (
    <div>
      {/* Card pequeno: tipo apenas como texto no bloco azul */}
      <div className="bg-blue-dark rounded-md shadow-md p-2">
        <div
          className="relative bg-gray-light rounded-md overflow-hidden border border-gray-300"
          style={{ paddingTop: '50%' }} 
        >
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src={props.imgSrc || placeholderIcon}
              alt={props.name || 'Imagem Placeholder'}
              className="w-full h-full object-contain cursor-pointer" 
              onClick={() => setIsModalOpen(true)} 
            />
          </div>
        </div>
        {/* Tipo como texto simples */}
        <div className="mt-2 text-white text-sm text-center font-semibold">
          {props.tipo}
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
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={() => setIsModalOpen(false)} 
        >
          <div className="relative bg-white rounded-md p-6 max-w-lg" onClick={e => e.stopPropagation()}>
            <img
              src={props.imgSrc || placeholderIcon}
              alt={props.name || 'Imagem Ampliada'}
              className="max-w-full max-h-96 rounded-md mx-auto"
            />
            <button
              className="absolute top-2 right-2 bg-white text-black rounded-full p-2"
              onClick={() => setIsModalOpen(false)}
            >
              ✕
            </button>
            <div className="mt-4 text-[#010131] text-sm">
              <div><strong>Nome:</strong> {props.name}</div>
              <div><strong>Tipo:</strong> {props.tipo}</div>
              {props.gravidade && <div><strong>Gravidade:</strong> {props.gravidade}</div>}
              {props.confianca !== undefined && props.confianca !== null && (
                <div><strong>Acurácia:</strong> {(props.confianca * 100).toFixed(1)}%</div>
              )}
              {coords && (
                <div className="mt-2">
                  <strong>Coordenadas:</strong>
                  <div>
                    x: {coords.x}, y: {coords.y}, largura: {coords.w}, altura: {coords.h}
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
