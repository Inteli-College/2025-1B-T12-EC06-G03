"use client";

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import ProjectRecent from '../components/ProjectRecent';
import ProjectAll from '../components/ProjectAll';

export default function ProjectPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [projects] = useState([
    { id: 1, name: "USP" },
    { id: 2, name: "IBM" },
    { id: 3, name: "Meta" },
    { id: 4, name: "Apontar" },
  ]);

  const [showModal, setShowModal] = useState(false);
  const [newProject, setNewProject] = useState({
    nome: "",
    cliente: "",
    descricao: "",
    status: "em andamento",
  });

  const navigate = useNavigate();

  const handleOpenProject = (projectName) => {
    navigate(`/relatorio?projeto=${projectName.toLowerCase()}`);
  };

  const handleCreateProject = () => {
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setNewProject({
      nome: "",
      cliente: "",
      descricao: "",
      status: "em andamento",
    });
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewProject({ ...newProject, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Projeto criado:", newProject);
    handleCloseModal();
  };

  const recentProjects = projects.slice(0, 4);
  const filteredProjects = projects.filter((project) =>
    project.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <main className="container mx-auto p-6">
      <ProjectRecent
        projects={recentProjects}
        onCreateProject={handleCreateProject}
        onViewReport={handleOpenProject}
      />

      <ProjectAll
        projects={filteredProjects}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        onViewReport={handleOpenProject}
      />

      {showModal && (
        <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-50">
          <div className="bg-white p-8 rounded-lg w-full max-w-2xl">
            <h2 className="text-2xl font-bold mb-4">Criar Novo Projeto</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block font-medium">Nome do Projeto</label>
                <input
                  type="text"
                  name="nome"
                  value={newProject.nome}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 p-2 rounded"
                />
              </div>

              <div>
                <label className="block font-medium">Cliente</label>
                <input
                  type="text"
                  name="cliente"
                  value={newProject.cliente}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 p-2 rounded"
                />
              </div>

              <div>
                <label className="block font-medium">Descrição</label>
                <textarea
                  name="descricao"
                  value={newProject.descricao}
                  onChange={handleInputChange}
                  required
                  className="w-full border border-gray-300 p-2 rounded h-24"
                />
              </div>


              <div className="flex justify-end gap-4 pt-4">
                <button
                  type="button"
                  onClick={handleCloseModal}
                  className="px-4 py-2 bg-gray-400 text-white rounded"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Criar Projeto
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </main>
  );
}
