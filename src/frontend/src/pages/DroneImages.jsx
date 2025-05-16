import React, { useState, useEffect } from 'react'
import placeholder from '../assets/placeholder-icon.svg'

const DroneImages = () => {
  const [selectedProject, setSelectedProject] = useState('')
  const [showModal, setShowModal] = useState(true)
  const [showConfirmModal, setShowConfirmModal] = useState(false)
  const [selectedFace, setSelectedFace] = useState('')
  const [projects] = useState(['USP', 'IBM', 'Meta', 'Apontar'])

  const handleSelectProject = (event) => {
    setSelectedProject(event.target.value)
  }

  const handleSelectFace = (event) => {
    const newProject = event.target.value
    setSelectedFace(newProject)
    setShowConfirmModal(true)
  }

  const handleCloseModal = () => {
    setShowModal(false)
  }

  const handleCloseConfirmModal = () => {
    setShowConfirmModal(false)
  }

  const handleConfirmSelection = () => {
    setShowConfirmModal(false)
    setSelectedProject(selectedFace)
  }

  useEffect(() => {
    if (showModal || showConfirmModal) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'auto'
    }
  }, [showModal, showConfirmModal])

  useEffect(() => {
    if (!selectedProject) {
      setShowModal(true)
    }
  }, [selectedProject])

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {showModal && (
        <div className="fixed inset-0 flex justify-center items-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg w-1/3">
            <h3 className="text-xl font-semibold mb-4">Selecione o projeto</h3>
            <select
              className="w-full p-2 border border-gray-300 rounded-md"
              value={selectedProject}
              onChange={handleSelectProject}
            >
              <option value="">Selecione um projeto</option>
              {projects.map((project) => (
                <option key={project} value={project}>
                  {project}
                </option>
              ))}
            </select>
            <div className="mt-4 text-right">
              <button
                className="bg-blue-darker text-white px-4 py-2 rounded-md"
                onClick={handleCloseModal}
              >
                Confirmar
              </button>
            </div>
          </div>
        </div>
      )}

      {showConfirmModal && (
        <div className="fixed inset-0 flex justify-center items-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg w-1/3">
            <h3 className="text-xl font-semibold mb-4">Confirmação de Projeto</h3>
            <p className="mb-4">
              Você selecionou o projeto <strong>{selectedFace}</strong>. As imagens serão enviadas para este projeto.
            </p>
            <div className="text-right">
              <button
                className="bg-gray-500 text-white px-4 py-2 rounded-md mr-2"
                onClick={handleCloseConfirmModal}
              >
                Cancelar
              </button>
              <button
                className="bg-blue-darker text-white px-4 py-2 rounded-md"
                onClick={handleConfirmSelection}
              >
                Confirmar
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="mb-8">
        <label className="block text-3xl font-semibold">Imagem do Drone</label>
        <select
          className="mt-2 bg-white p-2 border border-gray-300 rounded-md"
          value={selectedProject}
          onChange={handleSelectFace}
        >
          {projects.map((project) => (
            <option key={project} value={project}>
              {project}
            </option>
          ))}
        </select>
        <div className="mt-4 bg-gray-200 w-1/2 p-14 h-100 rounded-md flex justify-center items-center">
          <img src={placeholder} alt="Placeholder" className="max-h-full max-w-full object-contain" />
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Imagens Capturadas</h2>
        <div className="grid grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, index) => (
            <div
              key={index}
              className="w-34 h-34 border-8 border-blue-darker rounded-md flex justify-center items-center bg-gray-100"
            >
              <img src={placeholder} alt="Placeholder" className="max-h-full max-w-full object-contain p-4" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default DroneImages
