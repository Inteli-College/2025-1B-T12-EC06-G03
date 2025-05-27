import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

const VisualizarProjeto = () => {
  const [searchParams] = useSearchParams();
  const projetoId = searchParams.get("id");
  
  const [data, setData] = useState(null);
  const [formData, setFormData] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      if (!projetoId) {
        setError('ID do projeto não fornecido');
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch('http://localhost:8080/api/projeto/ViewProjeto', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json' 
          },
          body: JSON.stringify({ idProjeto: parseInt(projetoId) }),
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Transform backend data to match frontend expectations
        const transformedData = {
          message: {
            projeto: result.nome || '',
            responsaveis: result.responsaveisNomes || [],
            empresa: result.empresaNome || '',
            edificios: result.edificios?.map(edificio => ({
              nome: edificio.nome || '',
              localizacao: edificio.localizacao || '',
              tipo: edificio.tipo || '',
              pavimentos: edificio.pavimentos || '',
              ano_construcao: '' // This field might need to be added to your backend
            })) || [],
            descricao: result.descricao || '',
            logs_alteracoes: result.logs || []
          }
        };
        
        setData(transformedData);
        setFormData(transformedData.message);
        setError(null);
      } catch (err) {
        setError(err.message || 'Failed to fetch data');
        console.error('Error fetching project data:', err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, [projetoId]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleEdificioChange = (index, field, value) => {
    const updatedEdificios = [...formData.edificios];
    updatedEdificios[index] = {
      ...updatedEdificios[index],
      [field]: value
    };
    
    setFormData({
      ...formData,
      edificios: updatedEdificios
    });
  };

  const addResponsavel = () => {
    setFormData({
      ...formData,
      responsaveis: [...formData.responsaveis, ""]
    });
  };

  const removeResponsavel = (index) => {
    const updatedResponsaveis = [...formData.responsaveis];
    updatedResponsaveis.splice(index, 1);
    setFormData({
      ...formData,
      responsaveis: updatedResponsaveis
    });
  };

  const addEdificio = () => {
    setFormData({
      ...formData,
      edificios: [...formData.edificios, {
        nome: "",
        localizacao: "",
        tipo: "",
        pavimentos: "",
        ano_construcao: ""
      }]
    });
  };

  const removeEdificio = (index) => {
    const updatedEdificios = [...formData.edificios];
    updatedEdificios.splice(index, 1);
    setFormData({
      ...formData,
      edificios: updatedEdificios
    });
  };

  const handleSave = async () => {
    try {
      // Transform form data back to backend format
      const backendData = {
        nome: formData.projeto,
        responsaveisNomes: formData.responsaveis,
        empresaNome: formData.empresa,
        edificios: formData.edificios?.map(edificio => ({
          nome: edificio.nome,
          localizacao: edificio.localizacao,
          tipo: edificio.tipo,
          pavimentos: edificio.pavimentos
        })) || [],
        descricao: formData.descricao,
        logs: formData.logs_alteracoes
      };

      const response = await fetch('http://localhost:8080/api/projeto/UpdateViewProjeto', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idProjeto: parseInt(projetoId),
          viewProjetoResponseDTO: backendData
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const updatedData = await response.json();
      
      // Transform the response back to frontend format
      const transformedUpdatedData = {
        message: {
          projeto: updatedData.nome || '',
          responsaveis: updatedData.responsaveisNomes || [],
          empresa: updatedData.empresaNome || '',
          edificios: updatedData.edificios?.map(edificio => ({
            nome: edificio.nome || '',
            localizacao: edificio.localizacao || '',
            tipo: edificio.tipo || '',
            pavimentos: edificio.pavimentos || '',
            ano_construcao: ''
          })) || [],
          descricao: updatedData.descricao || '',
          logs_alteracoes: updatedData.logs || []
        }
      };
      
      setData(transformedUpdatedData);
      setIsEditing(false);
    } catch (err) {
      setError(err.message || 'Failed to save data');
      console.error('Error saving project data:', err);
    }
  };

  if (isLoading) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">Carregando...</div>;
  if (error) return <div className="text-center mt-10 text-red-500 font-lato text-2xl">Error: {error}</div>;
  if (!data) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">No data available</div>;

  return (
    <div className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      {isEditing ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-5xl font-lato text-[#010131]">{data.message.projeto}</h1>
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
            </div>
            {formData.responsaveis?.map((responsavel, index) => (
              <div key={index} className="flex items-center mb-2">
                <input
                  value={responsavel}
                  onChange={(e) => {
                    const updatedResponsaveis = [...formData.responsaveis];
                    updatedResponsaveis[index] = e.target.value;
                    setFormData({
                      ...formData,
                      responsaveis: updatedResponsaveis,
                    });
                  }}
                  className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                />
                <button
                  onClick={() => removeResponsavel(index)}
                  className="ml-2 p-2 bg-gray-medium text-white rounded font-lato"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
            <button 
              onClick={addResponsavel}
              className="flex items-center px-2 py-1 bg-dark-blue text-white rounded text-sm font-lato hover:bg-blue-darker"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-1">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
              Adicionar Responsável
            </button>
          </div>
          
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Empresa:</h3>
            <input
              name="empresa"
              value={formData.empresa || ''}
              onChange={handleChange}
              className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-2xl font-lato text-[#010131]">Edifícios:</h3>
            </div>
            {formData.edificios?.map((edificio, index) => (
              <div key={index} className="border rounded p-4 mb-4 relative">
                <button
                  onClick={() => removeEdificio(index)}
                  className="absolute top-2 right-2 p-1 bg-gray-medium text-white rounded font-lato"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Nome:</label>
                  <input
                    value={edificio.nome || ''}
                    onChange={(e) => handleEdificioChange(index, 'nome', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Localização:</label>
                  <input
                    value={edificio.localizacao || ''}
                    onChange={(e) => handleEdificioChange(index, 'localizacao', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Tipo:</label>
                  <input
                    value={edificio.tipo || ''}
                    onChange={(e) => handleEdificioChange(index, 'tipo', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Pavimentos:</label>
                  <input
                    value={edificio.pavimentos || ''}
                    onChange={(e) => handleEdificioChange(index, 'pavimentos', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Ano de Construção:</label>
                  <input
                    value={edificio.ano_construcao || ''}
                    onChange={(e) => handleEdificioChange(index, 'ano_construcao', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
              </div>
            ))}
            <button 
                onClick={addEdificio}
                className="flex items-center px-2 py-1 bg-dark-blue text-white rounded text-sm font-lato hover:bg-blue-darker"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-1">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
                Adicionar Edifício
              </button>
          </div>
          
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Descrição:</h3>
            <textarea
              name="descricao"
              value={formData.descricao || ''}
              onChange={handleChange}
              className="w-full border rounded p-2 h-24 text-1xl font-lato text-[#010131]"
            />
          </div>
          
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Logs de Alterações:</h3>
            <div className="border rounded p-4 mb-4 bg-gray-50">
              <p className="text-sm italic text-gray-500 font-lato">Os logs são atualizados automaticamente e não podem ser editados.</p>
              <ul className="list-disc pl-5 mt-2">
                {formData.logs_alteracoes?.map((log, index) => (
                  <li key={index} className="text-1xl font-lato text-[#010131]">{log}</li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="flex gap-4 justify-end">
            <button
              onClick={() => setIsEditing(false)}
              className="px-3 py-1 bg-gray-medium rounded text-lg font-lato text-white"
            >
              Cancelar
            </button>
            <button
              onClick={handleSave}
              className="px-3 py-1 bg-dark-blue text-white rounded text-lg font-lato hover:bg-blue-darker"
            >
              Salvar
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex items-center space-x-10">
            <h1 className="text-5xl font-lato text-[#010131]">{data.message.projeto}</h1>
            <svg 
              onClick={() => setIsEditing(true)} 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              strokeWidth={1.5} 
              stroke="currentColor" 
              className="w-8 h-8 cursor-pointer text-[#010131]"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125" />
            </svg>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
            <ul className="list-disc pl-5">
              {data.message.responsaveis.map((responsavel, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">{responsavel}</li>  
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">
              Empresa:
            </h3>
            <ul className="list-disc pl-5">
              <li className="text-1xl font-lato text-[#010131]">{data.message.empresa}</li>
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">
              Edifícios:
            </h3>
            <ul className="list-disc pl-5">
              {data.message.edificios.map((edificio, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">
                  <h4>{edificio.nome}</h4>
                  <ul className="list-disc pl-5 mt-1">
                    <li className="text-1xl font-lato text-[#010131]">Localização: {edificio.localizacao}</li>
                    <li className="text-1xl font-lato text-[#010131]">Tipo: {edificio.tipo}</li>
                    <li className="text-1xl font-lato text-[#010131]">Pavimentos: {edificio.pavimentos}</li>
                    {edificio.ano_construcao && (
                      <li className="text-1xl font-lato text-[#010131]">Ano de Construção: {edificio.ano_construcao}</li>
                    )}
                  </ul>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">
              Descrição:
            </h3>
            <ul className="list-disc pl-5">
              <li className="text-1xl font-lato text-[#010131]">{data.message.descricao}</li>
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">
              Logs de Alterações:
            </h3>
            <ul className="list-disc pl-5">
              {data.message.logs_alteracoes.map((log, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">{log}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualizarProjeto;