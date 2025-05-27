"use client";

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ProjectRecent from '../components/ProjectRecent';
import ProjectAll from '../components/ProjectAll';

export default function ProjectPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [showModal, setShowModal] = useState(false);
  const [newProject, setNewProject] = useState({
    nome: "",
    empresa: 1, // Default to first company, you might want to make this dynamic
    descricao: "",
    status: "EM_ANDAMENTO",
  });

  const navigate = useNavigate();

  // Fetch projects from backend
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await fetch('http://localhost:8080/api/projetos');
        if (!response.ok) {
          throw new Error('Failed to fetch projects');
        }
        const data = await response.json();
        // Transform backend data to match frontend expectations
        const transformedProjects = data.map(project => ({
          id: project.id,
          name: project.nome,
          status: project.status === 'EM_ANDAMENTO' ? 'em andamento' : 'finalizado',
          empresa: project.empresa,
          descricao: project.descricao,
          dataCriacao: project.dataCriacao,
          dataAtualizacao: project.dataAtualizacao
        }));
        setProjects(transformedProjects);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching projects:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  const handleOpenProject = (projectName) => {
    // Find project by name to get the ID
    const project = projects.find(p => p.name.toLowerCase() === projectName.toLowerCase());
    if (project) {
      navigate(`/relatorio?projeto=${projectName.toLowerCase()}&id=${project.id}`);
    }
  };

  const handleCreateProject = () => {
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setNewProject({
      nome: "",
      empresa: 1,
      descricao: "",
      status: "EM_ANDAMENTO",
    });
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewProject({ ...newProject, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8080/api/projetos', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newProject),
      });

      if (!response.ok) {
        throw new Error('Failed to create project');
      }

      const createdProject = await response.json();
      
      // Add the new project to the list
      const transformedProject = {
        id: createdProject.id,
        name: createdProject.nome,
        status: createdProject.status === 'EM_ANDAMENTO' ? 'em andamento' : 'finalizado',
        empresa: createdProject.empresa,
        descricao: createdProject.descricao,
        dataCriacao: createdProject.dataCriacao,
        dataAtualizacao: createdProject.dataAtualizacao
      };
      
      setProjects([...projects, transformedProject]);
      handleCloseModal();
    } catch (err) {
      console.error('Error creating project:', err);
      alert('Erro ao criar projeto: ' + err.message);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-8">
          <p className="text-lg">Carregando projetos...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-8">
          <p className="text-lg text-red-600">Erro ao carregar projetos: {error}</p>
        </div>
      </div>
    );
  }

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
                <label className="block font-medium">Empresa (ID)</label>
                <input
                  type="number"
                  name="empresa"
                  value={newProject.empresa}
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

              <div>
                <label className="block font-medium">Status</label>
                <select
                  name="status"
                  value={newProject.status}
                  onChange={handleInputChange}
                  className="w-full border border-gray-300 p-2 rounded"
                >
                  <option value="EM_ANDAMENTO">Em Andamento</option>
                  <option value="PLANEJAMENTO">Planejamento</option>
                  <option value="FINALIZADO">Finalizado</option>
                </select>
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
