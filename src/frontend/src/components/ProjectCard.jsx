const ProjectCard = ({ name }) => {
    return (
      <div className="bg-blue-300 rounded-md overflow-hidden">
        <div className="bg-white m-3 h-24 rounded-md"></div>
        <div className="p-3 flex justify-between items-center">
          <span className="font-medium">{name}</span>
          <button className="bg-[#050538] text-white text-sm px-3 py-1 rounded-md">Relatório</button>
        </div>
      </div>
    )
  }
  
  export default ProjectCard
  