"use client"

import ProjectCard from "./ProjectCard"

const ProjectRecent = ({ projects, onCreateProject }) => {
  return (
    <div className="mb-10">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold text-[#050538]">Projetos Recentes</h1>
        <button
          onClick={onCreateProject}
          className="bg-dark-blue text-white px-6 py-3 text-lg rounded-md hover:bg-blue-darker transition-colors"
        >
          Criar Projeto
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
        {projects.map((project) => (
          <ProjectCard key={project.id} name={project.name} />
        ))}
      </div>
    </div>
  )
}

export default ProjectRecent
