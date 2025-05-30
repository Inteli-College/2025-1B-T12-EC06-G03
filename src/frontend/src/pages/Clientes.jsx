import React, { useState, useEffect, useRef } from 'react';
import { Pencil, Trash2, Search, Plus } from 'lucide-react';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const Clientes = () => {
  const [clientes, setClientes] = useState([]);
  const [formulario, setFormulario] = useState({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
  const [editandoId, setEditandoId] = useState(null);
  const [busca, setBusca] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const [modalCadastrar, setModalCadastrar] = useState(false);
  const [modalEditar, setModalEditar] = useState(false);
  const [modalExcluir, setModalExcluir] = useState(false);
  const [clienteExcluirId, setClienteExcluirId] = useState(null);
  const [errors, setErrors] = useState({});

  const baseUrl = 'http://localhost:8080';
  const modalRef = useRef(null);

  useEffect(() => {
    async function fetchEmpresas() {
      try {
        const response = await fetch(`${baseUrl}/api/empresa/getEmpresas`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setClientes(data);
      } catch (err) {
        setError(err.message);
      }
    }
    fetchEmpresas();
  }, [baseUrl]);

  const limparFormulario = () => {
    setFormulario({ nome: '', cnpj: '', endereco: '', telefone: '', email: '' });
    setErrors({});
    setEditandoId(null);
    setError(null);
  };

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const validateCNPJ = (cnpj) => /^\d{14}$/.test(cnpj.replace(/\D/g, ''));

  const handleChange = (e) => {
    setFormulario({ ...formulario, [e.target.name]: e.target.value });
    setErrors(prev => ({ ...prev, [e.target.name]: '' }));
  };

  const validarFormulario = () => {
    const novosErros = {};
    if (!validateCNPJ(formulario.cnpj)) novosErros.cnpj = 'CNPJ deve conter 14 dígitos numéricos';
    if (!validateEmail(formulario.email)) novosErros.email = 'E-mail inválido';
    return novosErros;
  };

  const handleSubmit = async () => {
    const validationErrors = validarFormulario();
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (editandoId) {
        const res = await fetch(`${baseUrl}/api/empresa/update/${editandoId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formulario),
        });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const updatedCliente = await res.json();
        setClientes(prev => prev.map(c => (c.id === editandoId ? updatedCliente : c)));
        setModalEditar(false);
        toast.success('Cliente atualizado com sucesso');
      } else {
        const res = await fetch(`${baseUrl}/api/empresa/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formulario),
        });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const novoCliente = await res.json();
        setClientes(prev => [...prev, novoCliente]);
        setModalCadastrar(false);
        toast.success('Cliente cadastrado com sucesso');
      }
      limparFormulario();
    } catch (err) {
      setError(`Erro ao ${editandoId ? 'atualizar' : 'cadastrar'} cliente: ${err.message}`);
      toast.error(`Erro ao ${editandoId ? 'atualizar' : 'cadastrar'} cliente`);
    } finally {
      setLoading(false);
    }
  };

  const handleEditar = (cliente) => {
    setFormulario({ ...cliente });
    setEditandoId(cliente.id);
    setErrors({});
    setModalEditar(true);
  };

  const abrirModalExcluir = (id) => {
    setClienteExcluirId(id);
    setModalExcluir(true);
  };

  const handleExcluir = async () => {
    if (!clienteExcluirId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/api/empresa/delete/${clienteExcluirId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      setClientes(prev => prev.filter(c => c.id !== clienteExcluirId));
      setModalExcluir(false);
      setClienteExcluirId(null);
      toast.success('Cliente excluído com sucesso');
    } catch (err) {
      setError(`Erro ao excluir cliente: ${err.message}`);
      toast.error('Erro ao excluir cliente');
    } finally {
      setLoading(false);
    }
  };

  const fecharModalCadastrar = () => {
    setModalCadastrar(false);
    limparFormulario();
  };

  const fecharModalEditar = () => {
    setModalEditar(false);
    limparFormulario();
  };

  const fecharModalExcluir = () => {
    setModalExcluir(false);
    setClienteExcluirId(null);
  };

  const onOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      fecharModalCadastrar();
      fecharModalEditar();
      fecharModalExcluir();
      setError(null);
    }
  };

  const clientesFiltrados = clientes.filter(c =>
    c.nome.toLowerCase().includes(busca.toLowerCase()) ||
    c.cnpj.includes(busca) ||
    c.email.toLowerCase().includes(busca.toLowerCase())
  );

  if (error && clientes.length === 0) {
    return (
      <main className="container mx-auto p-6">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-red-600">Erro ao carregar clientes: {error}</div>
        </div>
      </main>
    );
  }

  const modalHeight = 480;
  const modalWidth = Math.floor(modalHeight * 1.15);

  return (
    <div className="max-w-4xl mx-auto p-8">
      <ToastContainer />
      <h1 className="text-4xl font-bold text-dark-blue mb-4">Clientes</h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="flex justify-end mb-4">
        <button
          onClick={() => setModalCadastrar(true)}
          className="pr-6 pl-4 py-4 rounded text-white font-semibold flex items-center gap-2 bg-blue-darker hover:bg-blue-dark transition"
          disabled={loading}
        >
          <Plus/> Cadastrar Cliente
        </button>
      </div>

      <div className="mb-4 relative">
        <Search className="absolute top-4 left-3 text-gray-400" size={18} />
        <input
          type="text"
          placeholder="Buscar cliente por nome, CNPJ ou e-mail"
          value={busca}
          onChange={e => setBusca(e.target.value)}
          className="w-full pl-10 p-3 border border-gray-300 rounded-md"
          disabled={loading}
        />
      </div>

      <div className="space-y-4">
        {clientesFiltrados.map(cliente => (
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
                className="text-black hover:text-blue-dark disabled:opacity-50 transition"
                disabled={loading}
                aria-label={`Editar cliente ${cliente.nome}`}
              >
                <Pencil />
              </button>
              <button
                onClick={() => abrirModalExcluir(cliente.id)}
                className="text-black hover:text-red-600 disabled:opacity-50 transition"
                disabled={loading}
                aria-label={`Excluir cliente ${cliente.nome}`}
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

      {modalCadastrar && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50"
          onClick={onOverlayClick}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="bg-white rounded-lg p-6 relative shadow-lg flex flex-col justify-between"
            style={{ width: modalWidth, height: modalHeight }}
            ref={modalRef}
          >
            <h2 className="text-dark-blue text-xl font-semibold mb-4">Cadastrar Cliente</h2>
            <form
              onSubmit={e => {
                e.preventDefault();
                handleSubmit();
              }}
              className="flex flex-col h-full justify-between"
            >
              <div>
                <input
                  type="text"
                  name="nome"
                  placeholder="Nome"
                  value={formulario.nome}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.nome ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.nome && <p className="text-red-600 text-sm mb-4">{errors.nome}</p>}

                <input
                  type="text"
                  name="cnpj"
                  placeholder="CNPJ"
                  value={formulario.cnpj}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.cnpj ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.cnpj && <p className="text-red-600 text-sm mb-4">{errors.cnpj}</p>}

                <input
                  type="text"
                  name="endereco"
                  placeholder="Endereço"
                  value={formulario.endereco}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.endereco ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.endereco && <p className="text-red-600 text-sm mb-4">{errors.endereco}</p>}

                <input
                  type="text"
                  name="telefone"
                  placeholder="Telefone"
                  value={formulario.telefone}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.telefone ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.telefone && <p className="text-red-600 text-sm mb-4">{errors.telefone}</p>}

                <input
                  type="email"
                  name="email"
                  placeholder="E-mail"
                  value={formulario.email}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.email ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.email && <p className="text-red-600 text-sm mb-4">{errors.email}</p>}
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="submit"
                  className="px-4 py-2 rounded text-white disabled:opacity-50 bg-blue-darker hover:bg-blue-dark transition"
                  disabled={loading}
                >
                  {loading ? 'Salvando...' : 'Cadastrar'}
                </button>
                <button
                  type="button"
                  onClick={fecharModalCadastrar}
                  className="px-4 py-2 rounded border border-gray-medium text-gray-medium hover:bg-gray-200 transition"
                  disabled={loading}
                >
                  Cancelar
                </button>
              </div>
            </form>
            <button
              className="absolute top-2 right-2 text-gray-medium hover:text-dark-blue text-2xl font-bold"
              onClick={fecharModalCadastrar}
              aria-label="Fechar modal"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {modalEditar && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50"
          onClick={onOverlayClick}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="bg-white rounded-lg p-6 relative shadow-lg flex flex-col justify-between"
            style={{ width: modalWidth, height: modalHeight }}
            ref={modalRef}
          >
            <h2 className="text-dark-blue text-xl font-semibold mb-4">Editar Cliente</h2>
            <form
              onSubmit={e => {
                e.preventDefault();
                handleSubmit();
              }}
              className="flex flex-col h-full justify-between"
            >
              <div>
                <input
                  type="text"
                  name="nome"
                  placeholder="Nome"
                  value={formulario.nome}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.nome ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.nome && <p className="text-red-600 text-sm mb-4">{errors.nome}</p>}

                <input
                  type="text"
                  name="cnpj"
                  placeholder="CNPJ"
                  value={formulario.cnpj}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.cnpj ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.cnpj && <p className="text-red-600 text-sm mb-4">{errors.cnpj}</p>}

                <input
                  type="text"
                  name="endereco"
                  placeholder="Endereço"
                  value={formulario.endereco}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.endereco ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.endereco && <p className="text-red-600 text-sm mb-4">{errors.endereco}</p>}

                <input
                  type="text"
                  name="telefone"
                  placeholder="Telefone"
                  value={formulario.telefone}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.telefone ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.telefone && <p className="text-red-600 text-sm mb-4">{errors.telefone}</p>}

                <input
                  type="email"
                  name="email"
                  placeholder="E-mail"
                  value={formulario.email}
                  onChange={handleChange}
                  className={`w-full border rounded px-3 py-2 mb-4 ${errors.email ? 'border-red-600' : 'border-gray-300'}`}
                  disabled={loading}
                  required
                />
                {errors.email && <p className="text-red-600 text-sm mb-4">{errors.email}</p>}
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="submit"
                  className="px-4 py-2 rounded text-white disabled:opacity-50 bg-blue-darker hover:bg-blue-dark transition"
                  disabled={loading}
                >
                  {loading ? 'Salvando...' : 'Salvar Alterações'}
                </button>
                <button
                  type="button"
                  onClick={fecharModalEditar}
                  className="px-4 py-2 rounded border border-gray-medium text-gray-medium hover:bg-gray-200 transition"
                  disabled={loading}
                >
                  Cancelar
                </button>
              </div>
            </form>
            <button
              className="absolute top-2 right-2 text-gray-medium hover:text-dark-blue text-2xl font-bold"
              onClick={fecharModalEditar}
              aria-label="Fechar modal"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {modalExcluir && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50"
          onClick={onOverlayClick}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="bg-white rounded-lg p-6 relative shadow-lg flex flex-col justify-between"
            style={{ width: modalWidth, height: modalHeight / 2 }}
            ref={modalRef}
          >
            <h2 className="text-dark-blue text-xl font-semibold mb-4 text-center">Confirmar Exclusão</h2>
            <p className="mb-4 text-center">Tem certeza que deseja excluir este cliente?</p>
            <div className="flex justify-center space-x-4">
              <button
                type="button"
                onClick={handleExcluir}
                className="px-6 py-2 rounded text-white bg-blue-darker hover:bg-blue-dark transition"
                disabled={loading}
              >
                {loading ? 'Excluindo...' : 'Sim, excluir'}
              </button>
              <button
                type="button"
                onClick={fecharModalExcluir}
                className="px-6 py-2 rounded border border-gray-medium text-gray-medium hover:bg-gray-200 transition"
                disabled={loading}
              >
                Cancelar
              </button>
            </div>
            <button
              className="absolute top-2 right-2 text-gray-medium hover:text-dark-blue text-2xl font-bold"
              onClick={fecharModalExcluir}
              aria-label="Fechar modal"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Clientes;
