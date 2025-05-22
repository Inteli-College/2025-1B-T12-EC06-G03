import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Pencil, Trash2 } from 'lucide-react';

const Edificios = () => {
  const [searchParams] = useSearchParams();
  const projetoAtivo = searchParams.get("projeto");

  const [edificios, setEdificios] = useState([
    {
      id: 1,
      nome: 'Bloco A',
      localizacao: 'Rua Alfa, 123',
      tipo: 'Administrativo',
      pavimentos: 3,
      projeto: 'usp',
      fachadas: [
        { area: 120, descricao: 'Fachada principal com revestimento em vidro' },
        { area: 80, descricao: 'Fachada lateral com concreto aparente' },
      ]
    },
  ]);

  const [formulario, setFormulario] = useState({ nome: '', localizacao: '', tipo: '', pavimentos: '', fachadas: [] });
  const [novaFachada, setNovaFachada] = useState({ area: '', descricao: '' });
  const [editandoId, setEditandoId] = useState(null);
  const [busca, setBusca] = useState('');

  const handleChange = (e) => {
    setFormulario({ ...formulario, [e.target.name]: e.target.value });
  };

  const handleFachadaChange = (e) => {
    setNovaFachada({ ...novaFachada, [e.target.name]: e.target.value });
  };

  const adicionarFachada = () => {
    if (!novaFachada.area || !novaFachada.descricao) return;
    setFormulario({ ...formulario, fachadas: [...formulario.fachadas, novaFachada] });
    setNovaFachada({ area: '', descricao: '' });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!projetoAtivo) return alert("Projeto não encontrado na URL");

    const novoEdificio = { ...formulario, id: editandoId || Date.now(), projeto: projetoAtivo };

    if (editandoId) {
      setEdificios((prev) => prev.map((ed) => (ed.id === editandoId ? novoEdificio : ed)));
      setEditandoId(null);
    } else {
      setEdificios((prev) => [...prev, novoEdificio]);
    }

    setFormulario({ nome: '', localizacao: '', tipo: '', pavimentos: '', fachadas: [] });
    setNovaFachada({ area: '', descricao: '' });
  };

  const handleEditar = (ed) => {
    setFormulario(ed);
    setEditandoId(ed.id);
  };

  const handleExcluir = (id) => {
    setEdificios((prev) => prev.filter((e) => e.id !== id));
  };

  const filtrados = edificios.filter(
    (e) => e.projeto === projetoAtivo && (
      e.nome.toLowerCase().includes(busca.toLowerCase()) ||
      e.localizacao.toLowerCase().includes(busca.toLowerCase())
    )
  );

  return (
    <div className="max-w-5xl mx-auto p-8">
      <h1 className="text-4xl font-bold text-[#050538] mb-6">Edifícios</h1>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Buscar por nome ou localização"
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-md"
        />
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-gray-100 p-6 rounded-md mb-8">
        <input name="nome" value={formulario.nome} onChange={handleChange} placeholder="Nome" required className="p-2 border border-gray-300 rounded" />
        <input name="localizacao" value={formulario.localizacao} onChange={handleChange} placeholder="Localização" required className="p-2 border border-gray-300 rounded" />
        <input name="tipo" value={formulario.tipo} onChange={handleChange} placeholder="Tipo" required className="p-2 border border-gray-300 rounded" />
        <input name="pavimentos" type="number" value={formulario.pavimentos} onChange={handleChange} placeholder="Pavimentos" required className="p-2 border border-gray-300 rounded" />

        <div className="col-span-full border-t border-gray-300 pt-4">
          <h3 className="text-lg font-semibold mb-2">Fachadas</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
            <input name="area" type="number" value={novaFachada.area} onChange={handleFachadaChange} placeholder="Área (m²)" className="p-2 border border-gray-300 rounded" />
            <input name="descricao" value={novaFachada.descricao} onChange={handleFachadaChange} placeholder="Descrição" className="p-2 border border-gray-300 rounded" />
            <button type="button" onClick={adicionarFachada} className="bg-blue-600 text-white rounded px-4 hover:bg-blue-700">
              Adicionar Fachada
            </button>
          </div>

          {formulario.fachadas.length > 0 && (
            <ul className="list-disc list-inside text-sm text-gray-700">
              {formulario.fachadas.map((f, i) => (
                <li key={i}>Área: {f.area} m² – {f.descricao}</li>
              ))}
            </ul>
          )}
        </div>

        <button type="submit" className="col-span-full bg-dark-blue text-white py-2 rounded hover:bg-blue-darker">
          {editandoId ? 'Salvar Alterações' : 'Cadastrar Edifício'}
        </button>
      </form>

      <div className="space-y-4">
        {filtrados.map((e) => (
          <div key={e.id} className="bg-white p-4 rounded shadow">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-lg font-semibold">{e.nome}</p>
                <p className="text-sm text-gray-600">Localização: {e.localizacao}</p>
                <p className="text-sm text-gray-600">Tipo: {e.tipo}</p>
                <p className="text-sm text-gray-600 mb-2">Pavimentos: {e.pavimentos}</p>
                {e.fachadas?.length > 0 && (
                  <div className="text-sm text-gray-800">
                    <p className="font-medium">Fachadas:</p>
                    <ul className="list-disc list-inside ml-4">
                      {e.fachadas.map((f, i) => (
                        <li key={i}>Área: {f.area} m² – {f.descricao}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <div className="flex gap-3 mt-1">
                <button onClick={() => handleEditar(e)} className="text-blue-600 hover:text-blue-800">
                  <Pencil />
                </button>
                <button onClick={() => handleExcluir(e.id)} className="text-red-600 hover:text-red-800">
                  <Trash2 />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Edificios;
