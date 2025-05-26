import React, { useState } from 'react';
import { Pencil, Trash2 } from 'lucide-react';

const Clientes = () => {
  const [clientes, setClientes] = useState([
    { id: 1, nome: 'Empresa A', cnpj: '00.000.000/0001-00', endereco: 'Rua A, 100', telefone: '(11) 1111-1111', email: 'empresaA@email.com' },
    { id: 2, nome: 'Empresa B', cnpj: '11.111.111/0001-11', endereco: 'Rua B, 200', telefone: '(11) 2222-2222', email: 'empresaB@email.com' }
  ]);

  const [formulario, setFormulario] = useState({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
  const [editandoId, setEditandoId] = useState(null);
  const [busca, setBusca] = useState('');

  const handleChange = (e) => {
    setFormulario({ ...formulario, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (editandoId) {
      setClientes((prev) =>
        prev.map((cliente) => (cliente.id === editandoId ? { ...formulario, id: cliente.id } : cliente))
      );
      setEditandoId(null);
    } else {
      const novoCliente = { ...formulario, id: Date.now() };
      setClientes((prev) => [...prev, novoCliente]);
    }
    setFormulario({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
  };

  const handleEditar = (cliente) => {
    setFormulario(cliente);
    setEditandoId(cliente.id);
  };

  const handleExcluir = (id) => {
    setClientes((prev) => prev.filter((cliente) => cliente.id !== id));
  };

  const clientesFiltrados = clientes.filter((c) =>
    c.nome.toLowerCase().includes(busca.toLowerCase()) ||
    c.cnpj.includes(busca) ||
    c.email.toLowerCase().includes(busca.toLowerCase())
  );

  return (
    <div className="max-w-4xl mx-auto p-8">
      <h1 className="text-4xl font-bold text-[#050538] mb-6">Clientes</h1>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Buscar cliente por nome, CNPJ ou e-mail"
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-md"
        />
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-gray-100 p-6 rounded-md mb-8">
        <input name="nome" value={formulario.nome} onChange={handleChange} placeholder="Nome" required className="p-2 border border-gray-300 rounded" />
        <input name="cnpj" value={formulario.cnpj} onChange={handleChange} placeholder="CNPJ" required className="p-2 border border-gray-300 rounded" />
        <input name="endereco" value={formulario.endereco} onChange={handleChange} placeholder="Endereço" required className="p-2 border border-gray-300 rounded" />
        <input name="telefone" value={formulario.telefone} onChange={handleChange} placeholder="Telefone" required className="p-2 border border-gray-300 rounded" />
        <input name="email" value={formulario.email} onChange={handleChange} placeholder="E-mail" required className="p-2 border border-gray-300 rounded" />
        <button type="submit" className="col-span-full bg-dark-blue text-white py-2 rounded hover:bg-blue-darker">
          {editandoId ? 'Salvar Alterações' : 'Cadastrar Cliente'}
        </button>
      </form>

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
              <button onClick={() => handleEditar(cliente)} className="text-blue-600 hover:text-blue-800">
                <Pencil />
              </button>
              <button onClick={() => handleExcluir(cliente.id)} className="text-red-600 hover:text-red-800">
                <Trash2 />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Clientes;
