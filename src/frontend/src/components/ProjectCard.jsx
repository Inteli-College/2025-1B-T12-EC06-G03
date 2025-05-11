const ProjectCard = ({ name }) => {
    return (
      <div className="bg-blue-300 rounded-md overflow-hidden shadow-md">
      <div className="bg-white m-3 h-36 rounded-md"></div>
      <div className="p-4 flex justify-between items-center">
        <span className="font-medium text-lg">{name}</span>
        <button className="bg-[#050538] text-white px-4 py-2 rounded-md">Relatório</button>
        </div>
      </div>
    )
  }
  
  export default ProjectCard
  