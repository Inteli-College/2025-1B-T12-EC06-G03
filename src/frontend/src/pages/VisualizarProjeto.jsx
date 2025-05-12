import React, { useEffect, useState } from 'react';

const VisualizarProjeto = () => {
  const [data, setData] = useState(null);
  const [formData, setFormData] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch('http://127.0.0.1:5000/teste');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        setData(result);
        setFormData(result);
        setError(null);
      } catch (err) {
        setError(err.message || 'Failed to fetch data');
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSave = async () => {
    await fetch('http://127.0.0.1:5000/teste/save', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });

    setData(formData);
    setIsEditing(false);
  };

  if (isLoading) return <div className="text-center mt-10">Carregando...</div>;
  if (error) return <div className="text-center mt-10 text-red-500">Error: {error}</div>;
  if (!data) return <div className="text-center mt-10">No data available</div>;

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-xl shadow-md">
      <div className=''>
        <h1>{data.message.projeto}</h1>
        <button
            onClick={() => setIsEditing(true)}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded"
          >
            Editar
          </button>
        </div>
      {isEditing ? (
        <div className="space-y-4">
          <input
            name="name"
            value={formData.responsaveis}
            onChange={handleChange}
            className="w-full border rounded p-2"
            placeholder="Responsáveis"
          />
          <div className="flex gap-4 justify-end">
            <button
              onClick={() => setIsEditing(false)}
              className="px-4 py-2 bg-gray-300 rounded"
            >
              Cancelar
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 text-white rounded"
            >
              Salvar
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4 font-lato text-[#010131]">
          <div>
            <h3>Responsáveis:</h3>
            {data.message.responsaveis.map((responsavel, index) => (
              <ul>
                <li>{responsavel}</li>
              </ul>
            ))}
          </div>
          <div>
            <h3>
              Empresa:
            </h3>
            <ul>
              <li>{data.message.empresa}</li>
            </ul>
          </div>
          <div>
            <h3>
              Edifícios:
            </h3>
            {data.message.edificios.map((edificio, index) => (
              <ul>
                <li>Nome: {edificio.nome}</li>
                <li>Localização: {edificio.localizacao}</li>
                <li>Tipo: {edificio.tipo}</li>
                <li>Pavimentos: {edificio.pavimentos}</li>
                <li>Ano de Construção: {edificio.ano_construcao}</li>
              </ul>
            ))}
          </div>
          <div>
            <h3>
              Descrição:
            </h3>
            <ul>
              <li>{data.message.descricao}</li>
            </ul>
          </div>
          <div>
            <h3>
              Logs de Alterações:
            </h3>
            {data.message.logs_alteracoes.map((log, index) => (
              <ul>
                <li>{log}</li>
              </ul>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualizarProjeto;