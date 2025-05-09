import ProjectElements from "./ProjectElements"
import ProjectSearch from "./ProjectSearch"

const ProjectAll = ({ projects, searchTerm, onSearchChange }) => {
  return (
    <div>
      <h1 className="text-3xl font-bold text-[#050538] mb-4">Todos os Projetos</h1>

      <ProjectSearch searchTerm={searchTerm} onSearchChange={onSearchChange} />

      <div className="flex flex-col gap-4">
        {projects.map((project) => (
          <ProjectElements key={project.id} name={project.name} />
        ))}
      </div>
    </div>
  )
}

export default ProjectAll
