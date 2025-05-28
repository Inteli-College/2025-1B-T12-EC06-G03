"use client";

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ProjectRecent from '../components/ProjectRecent';
import ProjectAll from '../components/ProjectAll';

export default function ProjectPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [projects, setProjects] = useState([]);
  const [empresas, setEmpresas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const baseUrl = import.meta.env?.VITE_BACKEND_BASE_URL || 'http://localhost:8080';

  const [showModal, setShowModal] = useState(false);
  const [newProject, setNewProject] = useState({
    nome: "",
    empresa: null,
    descricao: "",
    status: "EM_ANDAMENTO",
  });

  const navigate = useNavigate();

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch(`${baseUrl}/api/projetos`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setProjects(data);

        const res = await fetch(`${baseUrl}/api/empresa/getEmpresas`);
        const dados = await res.json();
        setEmpresas(dados);

      } catch (err) {
        console.error('Error fetching projects:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

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
      empresa: null,
      descricao: "",
      status: "EM_ANDAMENTO",
    });
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewProject({ ...newProject, [name]: value });
  };

  const handleEmpresaChange = (e) => {
    const { value } = e.target;
    setNewProject({
      ...newProject,
      empresa: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const empresaSelecionada = empresas.find(emp => emp.nome === newProject.empresa);
    
    if (!empresaSelecionada) {
      alert('Por favor, selecione uma empresa válida');
      return;
    }

    const projectData = {
      nome: newProject.nome,
      empresa: empresaSelecionada.id,
      descricao: newProject.descricao,
      status: newProject.status
    };

    const response = await fetch(`${baseUrl}/api/projetos`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(projectData)
    });

    if (!response.ok) {
      throw new Error(`Erro HTTP: ${response.status}`);
    }

    const createdProject = await response.json();
    console.log('Projeto criado com sucesso:', createdProject);

    setProjects(prevProjects => [...prevProjects, createdProject]);

    handleCloseModal();

    alert('Projeto criado com sucesso!');
  };

  const recentProjects = projects.slice(0, 4);
  const filteredProjects = projects.filter((project) =>
    project.nome.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <main className="container mx-auto p-6">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg">Carregando projetos...</div>
        </div>
      </main>
    );
  }

  if (error && projects.length === 0) {
    return (
      <main className="container mx-auto p-6">
        <div className="flex justify-center items-center h-64">
          <div className="text-lg text-red-600">
            Erro ao carregar projetos: {error}
          </div>
        </div>
      </main>
    );
  }

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
                <select
                  name="empresa"
                  value={newProject.empresa}
                  onChange={handleEmpresaChange}
                  required
                  className="w-full border border-gray-300 p-2 rounded"
                >
                  <option value="">Selecione uma empresa</option>
                  {empresas.map((empresa) => (
                    <option key={empresa.id} value={empresa.nome}>
                      {empresa.nome}
                    </option>
                  ))}
                </select>
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