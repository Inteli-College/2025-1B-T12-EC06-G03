import React, { useState, useEffect, useCallback } from 'react';
import { Pencil, Trash2, AlertCircle } from 'lucide-react';
import EdificioForm from '../components/EdificioForm';

const Edificios = () => {
  const getProjetoFromUrl = () => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get("projeto");
  };
  
  const projetoAtivo = getProjetoFromUrl();

  const [edificios, setEdificios] = useState([]);
  const [editandoEdificio, setEditandoEdificio] = useState(null);
  const [busca, setBusca] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_BASE_URL = 'http://localhost:8080/api/edificio';

  const loadEdificios = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const url = projetoAtivo 
        ? `${API_BASE_URL}/projeto-nome/${encodeURIComponent(projetoAtivo)}`
        : API_BASE_URL;
      
      const response = await fetch(url);
      
      if (response.status === 204) {
        // No content - lista vazia
        setEdificios([]);
      } else if (response.status === 404) {
        setError(`Projeto "${projetoAtivo}" não encontrado`);
        setEdificios([]);
      } else if (response.ok) {
        const data = await response.json();
        setEdificios(data || []);
      } else {
        throw new Error(`Error loading edificios: ${response.status}`);
      }
    } catch (err) {
      setError('Erro ao carregar edifícios: ' + err.message);
      console.error('Load error:', err);
    } finally {
      setLoading(false);
    }
  }, [projetoAtivo]);

  useEffect(() => {
    if (projetoAtivo) {
      loadEdificios();
    }
  }, [projetoAtivo, loadEdificios]);

  const handleFormSubmit = async (formData) => {
    if (!projetoAtivo) {
      setError("Projeto não encontrado na URL");
      return;
    }

    setLoading(true);
    setError('');

    try {
      let response;
      
      if (editandoEdificio) {
        const edificioData = {
          ...formData,
          projeto: editandoEdificio.projeto
        };
        response = await fetch(`${API_BASE_URL}/${editandoEdificio.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(edificioData)
        });
      } else {
        response = await fetch(`${API_BASE_URL}/projeto-nome/${encodeURIComponent(projetoAtivo)}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData)
        });
      }

      if (response.ok) {
        await loadEdificios();
        setEditandoEdificio(null);
      } else if (response.status === 404) {
        setError(`Projeto "${projetoAtivo}" não encontrado`);
      } else {
        throw new Error(`Error saving edificio: ${response.status}`);
      }
    } catch (err) {
      setError('Erro ao salvar edifício: ' + err.message);
      console.error('Save error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEditar = (edificio) => {
    setEditandoEdificio(edificio);
  };

  const handleCancelarEdicao = () => {
    setEditandoEdificio(null);
  };

  const handleExcluir = async (id) => {
    if (!window.confirm('Tem certeza que deseja excluir este edifício?')) {
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/${id}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadEdificios();
        // Se estava editando o edifício excluído, cancelar edição
        if (editandoEdificio && editandoEdificio.id === id) {
          setEditandoEdificio(null);
        }
      } else {
        throw new Error(`Error deleting edificio: ${response.status}`);
      }
    } catch (err) {
      setError('Erro ao excluir edifício: ' + err.message);
      console.error('Delete error:', err);
    } finally {
      setLoading(false);
    }
  };

  const filtrados = edificios.filter((e) =>
    e.nome?.toLowerCase().includes(busca.toLowerCase()) ||
    e.localizacao?.toLowerCase().includes(busca.toLowerCase())
  );

  if (!projetoAtivo) {
    return (
      <div className="max-w-5xl mx-auto p-8">
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
          <div className="flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <span>Projeto não especificado na URL. Adicione ?projeto=NOME_DO_PROJETO</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-8">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-4xl font-bold text-[#050538]">Edifícios</h1>
          <p className="text-gray-600 mt-1">Projeto: <span className="font-semibold">{projetoAtivo}</span></p>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <div className="flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <span>{error}</span>
          </div>
        </div>
      )}

      <div className="mb-6">
        <input
          type="text"
          placeholder="Buscar por nome ou localização"
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-md"
        />
      </div>

      {/* Formulário de Edifício usando o componente */}
      <EdificioForm
        initialData={editandoEdificio}
        onSubmit={handleFormSubmit}
        onCancel={handleCancelarEdicao}
        loading={loading}
        error=""
        isEditing={!!editandoEdificio}
        className="mb-8"
      />

      {loading && !error && (
        <div className="text-center py-4">
          <div className="text-gray-600">Carregando edifícios...</div>
        </div>
      )}

      <div className="space-y-4">
        {filtrados.length === 0 && !loading ? (
          <div className="text-center py-8 text-gray-500">
            {busca ? 'Nenhum edifício encontrado com os critérios de busca.' : 'Nenhum edifício cadastrado para este projeto.'}
          </div>
        ) : (
          filtrados.map((e) => (
            <div key={e.id} className="bg-white p-4 rounded shadow border">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <p className="text-lg font-semibold text-[#050538]">{e.nome}</p>
                  <p className="text-sm text-gray-600">Localização: {e.localizacao}</p>
                  <p className="text-sm text-gray-600">Tipo: {e.tipo}</p>
                  <p className="text-sm text-gray-600 mb-2">Pavimentos: {e.pavimentos}</p>
                  {e.fachadas && e.fachadas.length > 0 && (
                    <div className="text-sm text-gray-800">
                      <p className="font-medium">Fachadas:</p>
                      <ul className="list-disc list-inside ml-4 mt-1">
                        {e.fachadas.map((f, i) => (
                          <li key={i}>Área: {f.area} m² – {f.descricao}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div className="flex gap-2 ml-4">
                  <button 
                    onClick={() => handleEditar(e)} 
                    className="text-blue-600 hover:text-blue-800 p-1"
                    disabled={loading}
                    title="Editar"
                  >
                    <Pencil size={18} />
                  </button>
                  <button 
                    onClick={() => handleExcluir(e.id)} 
                    className="text-red-600 hover:text-red-800 p-1"
                    disabled={loading}
                    title="Excluir"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Edificios;