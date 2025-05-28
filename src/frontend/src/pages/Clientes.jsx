import React, { useState, useEffect} from 'react';
import { Pencil, Trash2 } from 'lucide-react';

const Clientes = () => {
  const [clientes, setClientes] = useState([]);
  const [formulario, setFormulario] = useState({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
  const [editandoId, setEditandoId] = useState(null);
  const [busca, setBusca] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const baseUrl = import.meta.env?.VITE_BACKEND_BASE_URL || 'http://localhost:8080';

  useEffect(() => {
    const fetchEmpresas = async () => {
      try {
        const response = await fetch(`${import.meta.env?.VITE_BACKEND_BASE_URL}/api/empresa/getEmpresas`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setClientes(data);

      } catch (err){
        console.error('Error:', err);
        setError(err.message);
      }
    };

    fetchEmpresas();
  }, []);

  const handleChange = (e) => {
    setFormulario({ ...formulario, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      if (editandoId) {
        // Update existing client
        const response = await fetch(`${baseUrl}/api/empresa/update/${editandoId}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formulario),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const updatedCliente = await response.json();
        setClientes((prev) =>
          prev.map((cliente) => (cliente.id === editandoId ? updatedCliente : cliente))
        );
        setEditandoId(null);
      } else {
        // Create new client
        const response = await fetch(`${baseUrl}/api/empresa/create`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formulario),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const novoCliente = await response.json();
        setClientes((prev) => [...prev, novoCliente]);
      }

      setFormulario({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
    } catch (err) {
      console.error('Error saving client:', err);
      setError(`Erro ao ${editandoId ? 'atualizar' : 'cadastrar'} cliente: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEditar = (cliente) => {
    setFormulario({
      nome: cliente.nome,
      cnpj: cliente.cnpj,
      endereco: cliente.endereco,
      telefone: cliente.telefone,
      email: cliente.email
    });
    setEditandoId(cliente.id);
  };

  const handleExcluir = async (id) => {
    if (!window.confirm('Tem certeza que deseja excluir este cliente?')) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${baseUrl}/api/empresa/delete/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setClientes((prev) => prev.filter((cliente) => cliente.id !== id));
    } catch (err) {
      console.error('Error deleting client:', err);
      setError(`Erro ao excluir cliente: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCancelarEdicao = () => {
    setEditandoId(null);
    setFormulario({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
    setError(null);
  };

  const clientesFiltrados = clientes.filter((c) =>
    c.nome.toLowerCase().includes(busca.toLowerCase()) ||
    c.cnpj.includes(busca) ||
    c.email.toLowerCase().includes(busca.toLowerCase())
  );

  if (error && clientes.length === 0) {
    return (
      <main className="container mx-auto p-6">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-red-600">
            Erro ao carregar clientes: {error}
          </div>
        </div>
      </main>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-4xl font-bold text-[#050538] mb-6">Clientes</h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-gray-100 p-6 rounded-md mb-8">
        <input 
          name="nome" 
          value={formulario.nome} 
          onChange={handleChange} 
          placeholder="Nome" 
          required 
          className="p-2 border border-gray-300 rounded"
          disabled={loading}
        />
        <input 
          name="cnpj" 
          value={formulario.cnpj} 
          onChange={handleChange} 
          placeholder="CNPJ" 
          required 
          className="p-2 border border-gray-300 rounded"
          disabled={loading}
        />
        <input 
          name="endereco" 
          value={formulario.endereco} 
          onChange={handleChange} 
          placeholder="Endereço" 
          required 
          className="p-2 border border-gray-300 rounded"
          disabled={loading}
        />
        <input 
          name="telefone" 
          value={formulario.telefone} 
          onChange={handleChange} 
          placeholder="Telefone" 
          required 
          className="p-2 border border-gray-300 rounded"
          disabled={loading}
        />
        <input 
          name="email" 
          value={formulario.email} 
          onChange={handleChange} 
          placeholder="E-mail" 
          required 
          className="p-2 border border-gray-300 rounded"
          disabled={loading}
        />
        <div className="col-span-full flex gap-2">
          <button 
            type="button"
            onClick={handleSubmit} 
            className="flex-1 bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? 'Salvando...' : (editandoId ? 'Salvar Alterações' : 'Cadastrar Cliente')}
          </button>
          {editandoId && (
            <button 
              type="button"
              onClick={handleCancelarEdicao}
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 disabled:opacity-50"
              disabled={loading}
            >
              Cancelar
            </button>
          )}
        </div>
      </div>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Buscar cliente por nome, CNPJ ou e-mail"
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-md"
          disabled={loading}
        />
      </div>

      <div className="space-y-4">
        {clientesFiltrados.map((cliente) => (
          <div key={cliente.id} className="bg-white p-4 rounded shadow flex justify-between items-center">
            <div>
              <p className="text-lg font-semibold">{cliente.nome}</p>
              <p className="text-sm text-gray-600">CNPJ: {cliente.cnpj}</p>
              <p className="text-sm text-gray-600">Endereço: {cliente.endereco}</p>
              <p className="text-sm text-gray-600">Telefone: {cliente.telefone}</p>
              <p className="text-sm text-gray-600">E-mail: {cliente.email}</p>
            </div>
            <div className="flex gap-3">
              <button 
                onClick={() => handleEditar(cliente)} 
                className="text-blue-600 hover:text-blue-800 disabled:opacity-50"
                disabled={loading}
              >
                <Pencil />
              </button>
              <button 
                onClick={() => handleExcluir(cliente.id)} 
                className="text-red-600 hover:text-red-800 disabled:opacity-50"
                disabled={loading}
              >
                <Trash2 />
              </button>
            </div>
          </div>
        ))}
      </div>

      {clientesFiltrados.length === 0 && clientes.length > 0 && (
        <div className="text-center text-gray-500 mt-8">
          Nenhum cliente encontrado com os critérios de busca.
        </div>
      )}
    </div>
  );
};

export default Clientes;