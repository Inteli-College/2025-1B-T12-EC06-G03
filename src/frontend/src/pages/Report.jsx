import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import html2pdf from 'html2pdf.js';
import { Pencil } from 'lucide-react';
import placeholder from '../assets/placeholder-icon.svg';

const Relatorios = () => {
  const [params] = useSearchParams();
  const projetoSelecionado = params.get("projeto")?.toLowerCase() || "usp";
  const [logsExtras, setLogsExtras] = useState([]);
  const [editando, setEditando] = useState(false);
  const [relatorioEditado, setRelatorioEditado] = useState("");
  const [statusProjeto, setStatusProjeto] = useState("em andamento");
  const [showModalEncerrar, setShowModalEncerrar] = useState(false);

  const initialData = {
    usp: {
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

  const data = initialData[projetoSelecionado];

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

  const handleEditar = () => {
    const conteudo = `
Responsáveis: ${data.responsaveis.join(", ")}
Empresa: ${data.empresa}
Edifícios: ${data.edificios.map(e => e.nome).join(", ")}
Descrição: ${data.descricao}
    `.trim();
    setRelatorioEditado(conteudo);
    setEditando(true);
  };

  const salvarEdicao = () => {
    const responsavel = prompt("Digite o nome do responsável pela alteração:");
    if (!responsavel) return;
    const descricaoMudanca = prompt("Descreva o que foi alterado:");
    if (!descricaoMudanca) return;

    const novaEntrada = `${new Date().toLocaleDateString()} - ${descricaoMudanca} (por ${responsavel})`;
    setLogsExtras((prev) => [...prev, novaEntrada]);
    setEditando(false);
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

  return (
    <div className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-5xl font-lato text-[#010131] flex items-center gap-4">
          {data.projeto}
          <Pencil
            size={28}
            className="text-gray-500 hover:text-gray-800 cursor-pointer"
            title="Editar Relatório"
            onClick={handleEditar}
          />
        </h1>
        <button
          onClick={exportarRelatorio}
          className="px-4 py-2 bg-dark-blue text-white rounded font-lato"
        >
          Exportar Relatório
        </button>
      </div>

      {editando && (
        <div className="mb-6">
          <textarea
            value={relatorioEditado}
            onChange={(e) => setRelatorioEditado(e.target.value)}
            className="w-full h-96 p-4 border border-gray-300 rounded mb-4"
          />
          <button
            onClick={salvarEdicao}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Salvar Alterações
          </button>
        </div>
      )}

      <div id="relatorio">
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
            {[...data.logs_alteracoes, ...logsExtras].map((log, i) => (
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

      {/* Modal de confirmação */}
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

export default Relatorios;
