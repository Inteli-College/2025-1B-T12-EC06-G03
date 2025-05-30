import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import placeholder from '../assets/placeholder-icon.svg';
import html2pdf from 'html2pdf.js';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Pencil } from 'lucide-react';

const Relatorios = () => {
  const [params] = useSearchParams();
  const projetoSelecionado = params.get("projeto") || "USP";

  const [fissuras, setFissuras] = useState([]);
  const [logsExtras, setLogsExtras] = useState([]);
  const [editando, setEditando] = useState(false);
  const [relatorioEditado, setRelatorioEditado] = useState("");
  const [statusProjeto, setStatusProjeto] = useState("em andamento");
  const [showModalEncerrar, setShowModalEncerrar] = useState(false);

  const dadosProjeto = {
    projeto: projetoSelecionado.toUpperCase(),
    responsaveis: ["Maria Lima", "Rafael Silva"],
    empresa: "USP",
    edificios: [{
      nome: "Prédio do LMPC Escola Politécnica da USP",
      localizacao: "Av. Professor Luciano Gualberto, travessa 3, n.º 158, São Paulo – SP",
      tipo: "Pesquisa e Ensino",
      pavimentos: 2,
      ano_construcao: "Estimado em 1980",
    }],
    descricao: "Este projeto tem como objetivo identificar fissuras na estrutura do prédio do LMPC...",
  };

  useEffect(() => {
    const fetchFissurasAprovadas = async () => {
      try {
        const response = await fetch(`http://localhost:8080/api/fissuras/aprovadas?projeto=${projetoSelecionado}`);
        const data = await response.json();
        setFissuras(data);
      } catch (error) {
        console.error("Erro ao buscar fissuras aprovadas:", error);
      }
    };

    fetchFissurasAprovadas();
  }, [projetoSelecionado]);

  const pieData = [
    { name: 'Fissuras Térmicas', value: fissuras.filter(f => f.tipo === "térmica").length },
    { name: 'Fissuras de Retração', value: fissuras.filter(f => f.tipo === "retração").length },
  ];
  const COLORS = ['#010131', '#75A1C0'];

  const exportarRelatorio = () => {
    const element = document.getElementById('relatorio');
    const options = {
      margin: 1,
      filename: `relatorio-${projetoSelecionado}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
    };
    html2pdf().set(options).from(element).save();
  };

  const handleEditar = () => {
    setRelatorioEditado(`
Responsáveis: ${dadosProjeto.responsaveis.join(", ")}
Empresa: ${dadosProjeto.empresa}
Edifícios: ${dadosProjeto.edificios.map(e => e.nome).join(", ")}
Descrição: ${dadosProjeto.descricao}
    `.trim());
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

  return (
    <div className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-5xl font-lato text-[#010131] flex items-center gap-4">
          {dadosProjeto.projeto}
          <Pencil size={28} className="text-gray-500 hover:text-gray-800 cursor-pointer" onClick={handleEditar} />
        </h1>
        <button onClick={exportarRelatorio} className="px-4 py-2 bg-dark-blue text-white rounded font-lato">
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
          <button onClick={salvarEdicao} className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
            Salvar Alterações
          </button>
        </div>
      )}

      <div id="relatorio">
        <h3 className="text-2xl font-lato text-[#010131]">Responsáveis:</h3>
        <ul className="list-disc pl-5 mb-4">{dadosProjeto.responsaveis.map((r, i) => <li key={i}>{r}</li>)}</ul>

        <h3 className="text-2xl font-lato text-[#010131]">Empresa:</h3>
        <p className="mb-4">{dadosProjeto.empresa}</p>

        <h3 className="text-2xl font-lato text-[#010131]">Edifícios:</h3>
        {dadosProjeto.edificios.map((ed, i) => (
          <div key={i} className="mb-4">
            <p><strong>Nome:</strong> {ed.nome}</p>
            <p><strong>Localização:</strong> {ed.localizacao}</p>
            <p><strong>Tipo:</strong> {ed.tipo}</p>
            <p><strong>Pavimentos:</strong> {ed.pavimentos}</p>
            <p><strong>Ano de Construção:</strong> {ed.ano_construcao}</p>
          </div>
        ))}

        <h3 className="text-2xl font-lato text-[#010131]">Descrição:</h3>
        <p className="mb-6">{dadosProjeto.descricao}</p>

        <h3 className="text-2xl font-lato text-[#010131] mb-2">Porcentagem de Fissuras:</h3>
        <div className="w-64 h-64 mx-auto">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <h3 className="text-2xl font-lato text-[#010131] mt-10">Imagens Aprovadas:</h3>
        <div className="grid grid-cols-2 gap-4 mt-4">
          {fissuras.map((fissura, index) => (
            <div key={index} className="border rounded p-4">
              <img
                src={`https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/${fissura.imagem?.caminhoArquivo}`}
                alt={`Fissura ${fissura.id}`}
                className="w-32 h-32 object-contain mx-auto mb-2 rounded"
                onError={(e) => { e.target.onerror = null; e.target.src = placeholder; }}
              />
              <p className="text-sm text-gray-800">{fissura.tipo} – Aprovado por {fissura.aprovadoPor || 'Desconhecido'}</p>
            </div>
          ))}
        </div>

        <div className="mt-8">
          <h3 className="text-2xl font-lato text-[#010131]">Logs de Alterações:</h3>
          <ul className="list-disc pl-5">
            {logsExtras.map((log, i) => <li key={i}>{log}</li>)}
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
              <button onClick={() => setShowModalEncerrar(false)} className="text-gray-600 hover:underline">
                Cancelar
              </button>
              <button onClick={handleConfirmEncerrar} className="bg-red-600 text-white px-4 py-2 rounded">
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
