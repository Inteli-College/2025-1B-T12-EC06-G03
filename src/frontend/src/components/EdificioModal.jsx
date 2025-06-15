import React from 'react';
import { X } from 'lucide-react';
import EdificioForm from './EdificioForm';

const EdificioModal = ({
  isOpen,
  onClose,
  initialData = null,
  onSubmit,
  loading = false,
  error = '',
  title = 'Edifício'
}) => {
  if (!isOpen) return null;

  const isEditing = !!initialData;

  const handleSubmit = (formData) => {
    if (onSubmit) {
      onSubmit(formData);
    }
  };

  const handleClose = () => {
    if (onClose && !loading) {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[95vh] overflow-y-auto">
        {/* Header do Modal */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-xl font-semibold text-gray-800">
            {isEditing ? `Editar ${title}` : `Novo ${title}`}
          </h2>
          <button
            onClick={handleClose}
            disabled={loading}
            className="text-gray-400 hover:text-gray-600 disabled:opacity-50"
          >
            <X size={24} />
          </button>
        </div>

        {/* Conteúdo do Modal */}
        <div className="p-6">
          <EdificioForm
            initialData={initialData}
            onSubmit={handleSubmit}
            onCancel={handleClose}
            loading={loading}
            error={error}
            isEditing={isEditing}
            className="bg-white border border-gray-200"
          />
        </div>
      </div>
    </div>
  );
};

export default EdificioModal;
