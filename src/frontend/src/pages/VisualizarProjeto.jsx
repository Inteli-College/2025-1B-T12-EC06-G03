import React, { useEffect, useState } from 'react';



const VisualizarProjeto = () => {
  const [data, setData] = useState(null);
  const [formData, setFormData] = useState({});
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    async function fetchData() {
      const response = await fetch('http://127.0.0.1:5000/teste');
      const result = await response.json();
      setData(result);
      setFormData(result);
    }

    fetchData();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSave = async () => {
    // await fetch('', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(formData),
    // });

    setData(formData);
    setIsEditing(false);
  };

  if (!data) return <div className="text-center mt-10">Carregando...</div>;

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-xl shadow-md">
        <h1>{data.projeto}</h1>
      {isEditing ? (
        <div className="space-y-4">
          <input
            name="name"
            value={formData.responsaveis}
            onChange={handleChange}
            className="w-full border rounded p-2"
            placeholder="Responsáveis"
          />
          <div className="flex gap-4 justify-end">
            <button
              onClick={() => setIsEditing(false)}
              className="px-4 py-2 bg-gray-300 rounded"
            >
              Cancelar
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 bg-blue-600 text-white rounded"
            >
              Salvar
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <p><strong>Responsáveis</strong> {data.responsaveis}</p>
          <button
            onClick={() => setIsEditing(true)}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded"
          >
            Editar
          </button>
        </div>
      )}
    </div>
  );
};

export default VisualizarProjeto;
