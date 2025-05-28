import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import ImagensCarregadas from '../components/ImagensCarregadas';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import html2pdf from 'html2pdf.js';

const VisualizarProjeto = () => {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [formData, setFormData] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pieData, setPieData] = useState([]);
  const [imagensProjeto, setImagensProjeto] = useState([]);
  const COLORS = ['#010131', '#75A1C0', '#0C668D', '#F7FCFE'];

  useEffect(() => {
    async function fetchData() {
      if (!id) {
        setError('ID do projeto não encontrado na URL');
        setIsLoading(false);
        return;
      }
      try {
        const response = await fetch(`http://localhost:8080/api/projeto/ViewProjeto`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ idProjeto: id }),
        });
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
  }, [id]);

  useEffect(() => {
    async function fetchPorcentagem() {
      try {
        const response = await fetch(`http://localhost:8080/api/fissura/porcentagem/${id}`);
        if (!response.ok) throw new Error('Erro ao buscar porcentagem de fissuras');
        const data = await response.json();
        // data.porcentagemPorTipo é um objeto { "Trinca fina": 25, ... }
        const pieArr = Object.entries(data.porcentagemPorTipo).map(([name, value]) => ({
          name,
          value
        }));
        setPieData(pieArr);
      } catch (err) {
        setError(err.message);
      }
    }
    if (id) fetchPorcentagem();
  }, [id]);

  useEffect(() => {
    async function fetchImagensProjeto() {
      try {
        const response = await fetch(`http://localhost:8080/api/images/${id}`);
        if (!response.ok) throw new Error('Erro ao buscar imagens do projeto');
        const imagens = await response.json();
        setImagensProjeto(imagens);
      } catch (err) {
        setError(err.message);
      }
    }
    if (id) fetchImagensProjeto();
  }, [id]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleEdificioChange = (index, field, value) => {
    const updatedEdificios = [...(formData.edificios || [])];
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
      responsaveis: [...(formData.responsaveis || []), ""]
    });
  };

  const removeResponsavel = (index) => {
    const updatedResponsaveis = [...(formData.responsaveis || [])];
    updatedResponsaveis.splice(index, 1);
    setFormData({
      ...formData,
      responsaveis: updatedResponsaveis
    });
  };

  const addEdificio = () => {
    setFormData({
      ...formData,
      edificios: [...(formData.edificios || []), {
        nome: "",
        localizacao: "",
        tipo: "",
        pavimentos: "",
        ano_construcao: ""
      }]
    });
  };

  const removeEdificio = (index) => {
    const updatedEdificios = [...(formData.edificios || [])];
    updatedEdificios.splice(index, 1);
    setFormData({
      ...formData,
      edificios: updatedEdificios
    });
  };

  const handleSave = async () => {
    if (!id) {
      setError('ID do projeto não encontrado');
      return;
    }
    try {
      const response = await fetch(`http://localhost:8080/api/projeto/UpdateViewProjeto`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idProjeto: id,
          viewProjetoResponseDTO: formData
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const updatedData = await response.json();
      setData(updatedData);
      setIsEditing(false);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to save data');
    }
  };

  const startEditing = () => {
    setFormData(data);
    setIsEditing(true);
  };

  const exportarRelatorio = () => {
    const element = document.getElementById('relatorio');
    const options = {
      margin: 1,
      filename: `relatorio-${data.projeto}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
    };
    html2pdf().set(options).from(element).save();
  };

  if (isLoading) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">Carregando...</div>;
  if (error) return <div className="text-center mt-10 text-red-500 font-lato text-2xl">Error: {error}</div>;
  if (!data) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">No data available</div>;

  return (
    <div className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-5xl font-lato text-[#010131] flex items-center gap-4">
          {data.projeto}
          <svg
            onClick={startEditing}
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-8 h-8 cursor-pointer text-[#010131]"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125" />
          </svg>
        </h1>
        <button
          onClick={exportarRelatorio}
          className="px-4 py-2 bg-dark-blue text-white rounded font-lato"
        >
          Exportar Relatório
        </button>
      </div>

      {isEditing ? (
        <div className="space-y-4">
          {/* Edição dos campos conectada ao backend */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
            </div>
            {(formData.responsaveis || []).map((responsavel, index) => (
              <div key={index} className="flex items-center mb-2">
                <input
                  value={responsavel}
                  onChange={(e) => {
                    const updatedResponsaveis = [...(formData.responsaveis || [])];
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
            {(formData.edificios || []).map((edificio, index) => (
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
                {(formData.logs_alteracoes || []).map((log, index) => (
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
        <div id="relatorio" className="space-y-6">
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
            <ul className="list-disc pl-5">
              {(data.responsaveis || []).map((responsavel, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">{responsavel}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Empresa:</h3>
            <p>{data.empresa}</p>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Edifícios:</h3>
            <ul className="list-disc pl-5">
              {(data.edificios || []).map((edificio, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">
                  <h4>{edificio.nome}</h4>
                  <ul className="list-disc pl-5 mt-1">
                    <li>Localização: {edificio.localizacao}</li>
                    <li>Tipo: {edificio.tipo}</li>
                    <li>Pavimentos: {edificio.pavimentos}</li>
                    <li>Ano de Construção: {edificio.ano_construcao}</li>
                  </ul>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Descrição:</h3>
            <p>{data.descricao}</p>
          </div>
          <div className="mt-8">
            <h3 className="text-2xl font-lato text-[#010131]">Porcentagem de Fissuras:</h3>
            <div className="w-80 h-80 mx-auto">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={110}
                    label
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="mt-8">
            <h3 className="text-2xl font-lato text-[#010131]">Imagens do Projeto:</h3>
            <div className="grid grid-cols-2 gap-4 mt-4">
              {imagensProjeto.map(imagem => {
                const SUPABASE_PROJECT_ID = "efinfalxxeaqfkvboewx"; 
                const SUPABASE_BUCKET = "img-projects";
                const url = `https://${SUPABASE_PROJECT_ID}.supabase.co/storage/v1/object/public/${SUPABASE_BUCKET}/${imagem.caminhoArquivo}`;
                return (
                  <ImagensCarregadas
                    key={imagem.id}
                    name={imagem.nomeArquivo}
                    imgSrc={url}
                  />
                );
              })}
            </div>
          </div>
          <div className="mt-8">
            <h3 className="text-2xl font-lato text-[#010131]">Logs de Alterações:</h3>
            <ul className="list-disc pl-5">
              {(data.logs_alteracoes || []).map((log, index) => (
                <li key={index} className="text-1xl font-lato text-[#010131]">{log}</li>
              ))}
            </ul>
            <div className="mt-4 flex items-center">
              {/* Adicione aqui o conteúdo desejado ou remova esta div se não for usar */}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VisualizarProjeto;