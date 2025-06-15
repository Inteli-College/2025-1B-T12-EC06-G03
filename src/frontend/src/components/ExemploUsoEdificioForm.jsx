// Exemplo de como usar o componente EdificioForm

import React, { useState } from 'react';
import EdificioForm from '../components/EdificioForm';
import EdificioModal from '../components/EdificioModal';

const ExemploUsoEdificioForm = () => {
  const [showModal, setShowModal] = useState(false);
  const [editingEdificio, setEditingEdificio] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Exemplo 1: Usando o formulário inline
  const handleFormSubmit = async (formData) => {
    setLoading(true);
    setError('');

    try {
      // Sua lógica de submissão aqui
      console.log('Dados do formulário:', formData);
      
      // Simular chamada API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Limpar formulário após sucesso
      setEditingEdificio(null);
    } catch (err) {
      setError('Erro ao salvar edifício');
    } finally {
      setLoading(false);
    }
  };

  // Exemplo 2: Usando o modal
  const handleModalSubmit = async (formData) => {
    setLoading(true);
    setError('');

    try {
      // Sua lógica de submissão aqui
      console.log('Dados do modal:', formData);
      
      // Simular chamada API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Fechar modal após sucesso
      setShowModal(false);
      setEditingEdificio(null);
    } catch (err) {
      setError('Erro ao salvar edifício');
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (edificio) => {
    setEditingEdificio(edificio);
    setShowModal(true);
  };

  const edificioExemplo = {
    id: 1,
    nome: 'Edifício Principal',
    localizacao: 'Centro da cidade',
    tipo: 'Comercial',
    pavimentos: 10,
    fachadas: [
      { area: 500, descricao: 'Fachada Norte' },
      { area: 450, descricao: 'Fachada Sul' }
    ]
  };

  return (
    <div className="max-w-6xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">Exemplo de Uso do EdificioForm</h1>

      {/* Exemplo 1: Formulário Inline */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">1. Formulário Inline</h2>
        <EdificioForm
          initialData={editingEdificio}
          onSubmit={handleFormSubmit}
          onCancel={() => setEditingEdificio(null)}
          loading={loading}
          error={error}
          isEditing={!!editingEdificio}
        />
      </div>

      {/* Exemplo 2: Botões para Modal */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">2. Formulário em Modal</h2>
        <div className="flex gap-4">
          <button
            onClick={() => {
              setEditingEdificio(null);
              setShowModal(true);
            }}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Novo Edifício (Modal)
          </button>
          <button
            onClick={() => handleEdit(edificioExemplo)}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            Editar Edifício (Modal)
          </button>
        </div>
      </div>

      {/* Modal */}
      <EdificioModal
        isOpen={showModal}
        onClose={() => {
          setShowModal(false);
          setEditingEdificio(null);
        }}
        initialData={editingEdificio}
        onSubmit={handleModalSubmit}
        loading={loading}
        error={error}
        title="Edifício"
      />

      {/* Exemplo de Props */}
      <div className="bg-gray-100 p-6 rounded-lg">
        <h3 className="text-lg font-semibold mb-4">Props do EdificioForm:</h3>
        <ul className="list-disc list-inside space-y-2 text-sm">
          <li><strong>initialData:</strong> Objeto com dados iniciais para edição (opcional)</li>
          <li><strong>onSubmit:</strong> Função chamada ao submeter o formulário</li>
          <li><strong>onCancel:</strong> Função chamada ao cancelar edição (opcional)</li>
          <li><strong>loading:</strong> Boolean para estado de carregamento</li>
          <li><strong>error:</strong> String com mensagem de erro</li>
          <li><strong>isEditing:</strong> Boolean indicando se está editando</li>
          <li><strong>className:</strong> Classes CSS adicionais</li>
        </ul>

        <h3 className="text-lg font-semibold mb-4 mt-6">Props do EdificioModal:</h3>
        <ul className="list-disc list-inside space-y-2 text-sm">
          <li><strong>isOpen:</strong> Boolean para controlar visibilidade do modal</li>
          <li><strong>onClose:</strong> Função para fechar o modal</li>
          <li><strong>title:</strong> Título do modal (padrão: "Edifício")</li>
          <li>+ todas as props do EdificioForm</li>
        </ul>
      </div>
    </div>
  );
};

export default ExemploUsoEdificioForm;
