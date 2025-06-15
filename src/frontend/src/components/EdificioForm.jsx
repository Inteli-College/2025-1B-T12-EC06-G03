import React, { useState, useEffect } from 'react';
import { Trash2, AlertCircle } from 'lucide-react';

const EdificioForm = ({
  initialData = null,
  onSubmit,
  onCancel,
  loading = false,
  error = '',
  isEditing = false,
  className = ''
}) => {
  const [formulario, setFormulario] = useState({
    nome: '',
    localizacao: '',
    tipo: '',
    pavimentos: '',
    fachadas: []
  });
  
  const [novaFachada, setNovaFachada] = useState({ area: '', descricao: '' });

  // Atualizar formulário quando initialData mudar
  useEffect(() => {
    if (initialData) {
      setFormulario({
        nome: initialData.nome || '',
        localizacao: initialData.localizacao || '',
        tipo: initialData.tipo || '',
        pavimentos: initialData.pavimentos?.toString() || '',
        fachadas: initialData.fachadas || []
      });
    } else {
      setFormulario({
        nome: '',
        localizacao: '',
        tipo: '',
        pavimentos: '',
        fachadas: []
      });
    }
  }, [initialData]);

  const handleChange = (e) => {
    setFormulario({ ...formulario, [e.target.name]: e.target.value });
  };

  const handleFachadaChange = (e) => {
    setNovaFachada({ ...novaFachada, [e.target.name]: e.target.value });
  };

  const adicionarFachada = () => {
    if (!novaFachada.area || !novaFachada.descricao) return;
    setFormulario({
      ...formulario,
      fachadas: [...formulario.fachadas, { ...novaFachada, area: Number(novaFachada.area) }]
    });
    setNovaFachada({ area: '', descricao: '' });
  };

  const removerFachada = (index) => {
    setFormulario({
      ...formulario,
      fachadas: formulario.fachadas.filter((_, i) => i !== index)
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSubmit) {
      onSubmit({
        ...formulario,
        pavimentos: Number(formulario.pavimentos)
      });
    }
  };

  const handleCancel = () => {
    setFormulario({
      nome: '',
      localizacao: '',
      tipo: '',
      pavimentos: '',
      fachadas: []
    });
    setNovaFachada({ area: '', descricao: '' });
    if (onCancel) {
      onCancel();
    }
  };

  return (
    <div className={`bg-gray-100 p-6 rounded-md ${className}`}>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <div className="flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <span>{error}</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
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
            name="localizacao"
            value={formulario.localizacao}
            onChange={handleChange}
            placeholder="Localização"
            required
            className="p-2 border border-gray-300 rounded"
            disabled={loading}
          />
          <input
            name="tipo"
            value={formulario.tipo}
            onChange={handleChange}
            placeholder="Tipo"
            required
            className="p-2 border border-gray-300 rounded"
            disabled={loading}
          />
          <input
            name="pavimentos"
            type="number"
            value={formulario.pavimentos}
            onChange={handleChange}
            placeholder="Pavimentos"
            required
            min="1"
            className="p-2 border border-gray-300 rounded"
            disabled={loading}
          />
        </div>

        {/* Seção de Fachadas */}
        <div className="border-t border-gray-300 pt-4 mb-6">
          <h3 className="text-lg font-semibold mb-2">Fachadas</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
            <input
              name="area"
              type="number"
              value={novaFachada.area}
              onChange={handleFachadaChange}
              placeholder="Área (m²)"
              min="0"
              step="0.01"
              className="p-2 border border-gray-300 rounded"
              disabled={loading}
            />
            <input
              name="descricao"
              value={novaFachada.descricao}
              onChange={handleFachadaChange}
              placeholder="Descrição"
              className="p-2 border border-gray-300 rounded"
              disabled={loading}
            />
            <button
              type="button"
              onClick={adicionarFachada}
              className="bg-blue-600 text-white rounded px-4 hover:bg-blue-700 disabled:opacity-50"
              disabled={loading || !novaFachada.area || !novaFachada.descricao}
            >
              Adicionar Fachada
            </button>
          </div>

          {formulario.fachadas.length > 0 && (
            <div className="bg-white p-3 rounded border">
              <p className="font-medium mb-2">Fachadas adicionadas:</p>
              <ul className="space-y-1">
                {formulario.fachadas.map((f, i) => (
                  <li key={i} className="flex justify-between items-center text-sm bg-gray-50 p-2 rounded">
                    <span>Área: {f.area} m² – {f.descricao}</span>
                    <button
                      type="button"
                      onClick={() => removerFachada(i)}
                      className="text-red-600 hover:text-red-800 ml-2"
                      disabled={loading}
                    >
                      <Trash2 size={16} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Botões de Ação */}
        <div className="flex gap-3">
          <button
            type="submit"
            className="flex-1 bg-blue-800 text-white py-2 rounded hover:bg-blue-900 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? 'Salvando...' : (isEditing ? 'Salvar Alterações' : 'Cadastrar Edifício')}
          </button>
          {isEditing && (
            <button
              type="button"
              onClick={handleCancel}
              className="px-6 bg-gray-500 text-white py-2 rounded hover:bg-gray-600 disabled:opacity-50"
              disabled={loading}
            >
              Cancelar
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default EdificioForm;
