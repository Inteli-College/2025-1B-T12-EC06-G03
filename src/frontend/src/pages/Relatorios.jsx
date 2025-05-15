import React, { useState } from 'react';
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';
import html2pdf from 'html2pdf.js';

const Relatorios = () => {
  const initialData = {
    message: {
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
        { id: 1, imagem: 'https://via.placeholder.com/150', descricao: 'Fissura na fachada leste, próximo à janela.' },
        { id: 2, imagem: 'https://via.placeholder.com/150', descricao: 'Fissura na base da coluna principal.' },
      ],
      porcentagemFissuras: {
        termica: 60,
        retracao: 40,
      },
    },
  };

  const [data] = useState(initialData);

  const exportarRelatorio = () => {
    const element = document.getElementById('relatorio');
    const options = {
      margin: 1,
      filename: 'relatorio.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
    };
    html2pdf().set(options).from(element).save();
  };

  // Dados para o gráfico de pizza do recharts
  const pieData = [
    { name: 'Fissuras Térmicas', value: data.message.porcentagemFissuras.termica },
    { name: 'Fissuras de Retração', value: data.message.porcentagemFissuras.retracao },
  ];
  const COLORS = ['#010131', '#75A1C0'];

  return (
    <div id="relatorio" className="max-w-3xl ml-14 mt-14 p-6 bg-white font-lato text-dark-blue">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-5xl font-lato text-[#010131]">{data.message.projeto}</h1>
        <button
          onClick={exportarRelatorio}
          className="px-4 py-2 bg-dark-blue text-white rounded font-lato"
        >
          Exportar Relatório
        </button>
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
        <h3 className="text-2xl font-lato text-[#010131]">Empresa:</h3>
        <p className="text-1xl font-lato text-[#010131]">{data.message.empresa}</p>
      </div>

      <div>
        <h3 className="text-2xl font-lato text-[#010131]">Edifícios:</h3>
        <ul className="list-disc pl-5">
          {data.message.edificios.map((edificio, index) => (
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
        <p className="text-1xl font-lato text-[#010131]">{data.message.descricao}</p>
      </div>

      <div>
        <h3 className="text-2xl font-lato text-[#010131]">Logs de Alterações:</h3>
        <ul className="list-disc pl-5">
          {data.message.logs_alteracoes.map((log, index) => (
            <li key={index} className="text-1xl font-lato text-[#010131]">{log}</li>
          ))}
        </ul>
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
          {data.message.fissuras.map((fissura) => (
            <div key={fissura.id} className="border rounded p-4">
              <img src={fissura.imagem} alt={`Fissura ${fissura.id}`} className="w-full h-32 object-cover rounded mb-2" />
              <p className="text-1xl font-lato text-[#010131]">{fissura.descricao}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Relatorios;