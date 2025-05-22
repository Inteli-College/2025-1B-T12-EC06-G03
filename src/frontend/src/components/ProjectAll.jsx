"use client";

import ProjectElement from "./ProjectElement";
import ProjectSearch from "./ProjectSearch";

const ProjectAll = ({ projects, searchTerm, onSearchChange, onViewReport }) => {
  return (
    <div>
      <h1 className="text-4xl font-bold text-[#050538] mb-6">Todos os Projetos</h1>

      <ProjectSearch searchTerm={searchTerm} onSearchChange={onSearchChange} />

      <div className="flex flex-col gap-6">
        {projects.map((project) => (
          <div key={project.id} className="bg-white p-4 rounded shadow">
            <ProjectElement
              name={project.name}
              onViewReport={onViewReport}
            />
            <div className="mt-2">
              <span className={`text-sm font-semibold px-3 py-1 rounded ${project.status === 'finalizado' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                {project.status === 'finalizado' ? 'Finalizado' : 'Em Andamento'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProjectAll;
