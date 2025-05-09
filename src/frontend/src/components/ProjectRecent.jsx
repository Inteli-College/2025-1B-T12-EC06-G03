"use client"

import ProjectCard from "./ProjectCard"

const ProjectRecent = ({ projects, onCreateProject }) => {
  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-[#050538]">Projetos Recentes</h1>
        <button
          onClick={onCreateProject}
          className="bg-[#050538] text-white px-4 py-2 rounded-md hover:bg-[#0a0a4a] transition-colors"
        >
          Criar Projeto
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
        {projects.map((project) => (
          <ProjectCard key={project.id} name={project.name} />
        ))}
      </div>
    </div>
  )
}

export default ProjectRecent
