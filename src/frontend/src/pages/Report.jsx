import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import html2pdf from 'html2pdf.js'; 
import { Pencil, X, Plus } from 'lucide-react'; 
import placeholder from '../assets/placeholder-icon.svg';

const Report = () => {
  const [params] = useSearchParams();
  const projetoSelecionado = params.get("projeto")?.toLowerCase() || "usp";
  const [isEditingStructural, setIsEditingStructural] = useState(false); 
  const [relatorioEditado, setRelatorioEditado] = useState(""); 
  const [statusProjeto, setStatusProjeto] = useState("em andamento");
  const [showModalEncerrar, setShowModalEncerrar] = useState(false);

  const initialData = {
    usp: {
      id: 1, 
      projeto: "USP",
      responsaveis: ["Maria Lima", "Rafael Silva"],
      empresa: "USP",
      edificios: [{
        nome: "Prédio do LMPC Escola Politécnica da USP",
        localizacao: "Av. Professor Luciano Gualberto, travessa 3, n.º 158, São Paulo – SP",
        tipo: "Pesquisa e Ensino",
        pavimentos: 2,
        ano_construcao: "Estimado em 1980", 
      }],
      descricao: "Este projeto tem como objetivo identificar fissuras na estrutura do prédio do LMPC, localizado na Escola Politécnica da USP. Utilizando imagens capturadas por drone, o sistema analisa as fachadas do edifício para detectar possíveis falhas estruturais.",
      logs_alteracoes: [
        "06/05/2025 - Upload da Imagem Captura01.png",
        "05/05/2025 - Análise da Imagem Upload03.png feita"
      ],
      fissuras: [
        { id: 1, imagem: placeholder, descricao: 'Fissura na fachada leste, próximo à janela.' },
        { id: 2, imagem: placeholder, descricao: 'Fissura na base da coluna principal.' },
      ],
      porcentagemFissuras: {
        termica: 60,
        retracao: 40,
      },
    }
  };

  const [data, setData] = useState(initialData[projetoSelecionado]);
  const [formData, setFormData] = useState(data); 
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false); 

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

  const handleStructuralChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleEdificioStructuralChange = (index, field, value) => {
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

  const handleSaveStructural = async () => {
    setIsLoading(true);
    try {
      const viewProjetoResponseDTO = {
        projeto: formData.projeto,
        responsaveis: formData.responsaveis,
        empresa: formData.empresa,
        edificios: formData.edificios.map(ed => ({
          nome: ed.nome,
          localizacao: ed.localizacao,
          tipo: ed.tipo,
          pavimentos: parseInt(ed.pavimentos) || 0, 
        })),
        descricao: formData.descricao,
      };

      const requestBody = {
        idProjeto: data.id,
        viewProjetoResponseDTO: viewProjetoResponseDTO,
      };

      const response = await fetch('http://127.0.0.1:8080/api/projeto/UpdateViewProjeto', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json(); 
      setData(result); 
      setIsEditingStructural(false); 
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to save data');
      console.error("Erro ao salvar dados estruturais:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelStructuralEdit = () => {
    setFormData(data);
    setIsEditingStructural(false);
  };

  const handleConfirmEncerrar = () => {
    setStatusProjeto("finalizado");
    setShowModalEncerrar(false);
  };

  const pieData = [
    { name: 'Fissuras Térmicas', value: data.porcentagemFissuras.termica },
    { name: 'Fissuras de Retração', value: data.porcentagemFissuras.retracao },
  ];
  const COLORS = ['#010131', '#75A1C0'];

  if (isLoading) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">Carregando...</div>;
  if (error) return <div className="text-center mt-10 text-red-500 font-lato text-2xl">Error: {error}</div>;
  if (!data) return <div className="text-center mt-10 font-lato text-[#010131] text-2xl">No data available</div>;

  return (
    <div className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-5xl font-lato text-[#010131] flex items-center gap-4">
          {data.projeto}
          <Pencil
            size={28}
            className="text-gray-500 hover:text-gray-800 cursor-pointer"
            title="Editar Detalhes do Projeto"
            onClick={() => setIsEditingStructural(true)}
          />
        </h1>
        <button
          onClick={exportarRelatorio}
          className="px-4 py-2 bg-dark-blue text-white rounded font-lato"
        >
          Exportar Relatório
        </button>
      </div>

      {isEditingStructural && (
        <div className="space-y-4 mb-8 p-6 border rounded-lg bg-gray-50">
          <h2 className="text-3xl font-lato text-[#010131] mb-4">Editar Detalhes do Projeto</h2>

          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Nome do Projeto:</h3>
            <input
              name="projeto"
              value={formData.projeto || ''}
              onChange={handleStructuralChange}
              className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
            />
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
                  className="ml-2 p-2 bg-gray-medium text-white rounded font-lato hover:bg-red-500"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
            <button
              onClick={addResponsavel}
              className="flex items-center px-2 py-1 bg-dark-blue text-white rounded text-sm font-lato hover:bg-blue-darker mt-2"
            >
              <Plus size={16} className="mr-1" />
              Adicionar Responsável
            </button>
          </div>

          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Empresa:</h3>
            <input
              name="empresa"
              value={formData.empresa || ''}
              onChange={handleStructuralChange}
              className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-2xl font-lato text-[#010131]">Edifícios:</h3>
            </div>
            {formData.edificios?.map((edificio, index) => (
              <div key={index} className="border rounded p-4 mb-4 relative bg-white">
                <button
                  onClick={() => removeEdificio(index)}
                  className="absolute top-2 right-2 p-1 bg-gray-medium text-white rounded font-lato hover:bg-red-500"
                >
                  <X size={16} />
                </button>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Nome:</label>
                  <input
                    value={edificio.nome || ''}
                    onChange={(e) => handleEdificioStructuralChange(index, 'nome', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Localização:</label>
                  <input
                    value={edificio.localizacao || ''}
                    onChange={(e) => handleEdificioStructuralChange(index, 'localizacao', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Tipo:</label>
                  <input
                    value={edificio.tipo || ''}
                    onChange={(e) => handleEdificioStructuralChange(index, 'tipo', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                <div className="mb-2">
                  <label className="block text-sm font-lato text-[#010131]">Pavimentos:</label>
                  <input
                    type="number" 
                    value={edificio.pavimentos || ''}
                    onChange={(e) => handleEdificioStructuralChange(index, 'pavimentos', e.target.value)}
                    className="w-full border rounded p-2 text-1xl font-lato text-[#010131]"
                  />
                </div>
                {edificio.ano_construcao && (
                  <div className="mb-2">
                    <label className="block text-sm font-lato text-[#010131]">Ano de Construção (Não Editável):</label>
                    <p className="w-full border rounded p-2 text-1xl font-lato text-[#010131] bg-gray-100">
                      {edificio.ano_construcao}
                    </p>
                  </div>
                )}
              </div>
            ))}
            <button
              onClick={addEdificio}
              className="flex items-center px-2 py-1 bg-dark-blue text-white rounded text-sm font-lato hover:bg-blue-darker mt-2"
            >
              <Plus size={16} className="mr-1" />
              Adicionar Edifício
            </button>
          </div>

          <div>
            <h3 className="text-2xl font-lato text-[#010131]">Descrição:</h3>
            <textarea
              name="descricao"
              value={formData.descricao || ''}
              onChange={handleStructuralChange}
              className="w-full border rounded p-2 h-24 text-1xl font-lato text-[#010131]"
            />
          </div>

          <div className="flex gap-4 justify-end">
            <button
              onClick={handleCancelStructuralEdit}
              className="px-3 py-1 bg-gray-medium rounded text-lg font-lato text-white hover:bg-gray-600"
            >
              Cancelar
            </button>
            <button
              onClick={handleSaveStructural}
              className="px-3 py-1 bg-dark-blue text-white rounded text-lg font-lato hover:bg-blue-darker"
            >
              Salvar
            </button>
          </div>
        </div>
      )}

      <div id="relatorio" className={isEditingStructural ? 'hidden' : ''}>
        <div>
          <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
          <ul className="list-disc pl-5">
            {data.responsaveis.map((r, i) => (
              <li key={i} className="text-1xl">{r}</li>
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
            {data.edificios.map((edificio, i) => (
              <li key={i} className="text-1xl">
                <h4>{edificio.nome}</h4>
                <ul className="list-disc pl-5 mt-1">
                  <li>Localização: {edificio.localizacao}</li>
                  <li>Tipo: {edificio.tipo}</li>
                  <li>Pavimentos: {edificio.pavimentos}</li>
                  {edificio.ano_construcao && (
                    <li>Ano de Construção: {edificio.ano_construcao}</li>
                  )}
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
          <div className="w-64 h-64 mx-auto">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
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
          <h3 className="text-2xl font-lato text-[#010131]">Imagens de Fissuras:</h3>
          <div className="grid grid-cols-2 gap-4 mt-4">
            {data.fissuras.map((f) => (
              <div key={f.id} className="border rounded p-4">
                <img
                  src={f.imagem}
                  alt={`Fissura ${f.id}`}
                  className="w-32 h-32 object-contain mx-auto rounded mb-2"
                  onError={(e) => { e.target.onerror = null; e.target.src = placeholder; }}
                />
                <p>{f.descricao}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-8">
          <h3 className="text-2xl font-lato text-[#010131]">Logs de Alterações:</h3>
          <ul className="list-disc pl-5">
            {data.logs_alteracoes.map((log, i) => (
              <li key={i} className="text-1xl">{log}</li>
            ))}
          </ul>

          <div className="mt-4 flex items-center gap-4">
            <span className={`text-sm font-semibold px-3 py-1 rounded ${statusProjeto === 'finalizado' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
              {statusProjeto === 'finalizado' ? 'Finalizado' : 'Em Andamento'}
            </span>
            {statusProjeto === 'em andamento' && (
              <button
                onClick={() => setShowModalEncerrar(true)}
                className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
              >
                Encerrar Projeto
              </button>
            )}
          </div>
        </div>
      </div>

      {showModalEncerrar && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white rounded p-6 max-w-md w-full shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Encerrar Projeto</h2>
            <p className="mb-4">Tem certeza de que deseja encerrar este projeto? Essa ação não pode ser desfeita.</p>
            <div className="flex justify-end gap-4">
              <button
                onClick={() => setShowModalEncerrar(false)}
                className="px-4 py-2 text-gray-600 hover:underline"
              >
                Cancelar
              </button>
              <button
                onClick={handleConfirmEncerrar}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Encerrar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Report;