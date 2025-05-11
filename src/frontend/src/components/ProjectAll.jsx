import ProjectElement from "./ProjectElement"
import ProjectSearch from "./ProjectSearch"

const ProjectAll = ({ projects, searchTerm, onSearchChange }) => {
  return (
    <div>
      <h1 className="text-4xl font-bold text-[#050538] mb-6">Todos os Projetos</h1>

      <ProjectSearch searchTerm={searchTerm} onSearchChange={onSearchChange} />

      <div className="flex flex-col gap-6">
        {projects.map((project) => (
          <ProjectElement key={project.id} name={project.name} />
        ))}
      </div>
    </div>
  )
}

export default ProjectAll
