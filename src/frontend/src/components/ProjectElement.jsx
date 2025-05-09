const ProjectElements = ({ name }) => {
    return (
      <div className="bg-blue-300 rounded-md p-4 flex justify-between items-center">
        <span className="font-medium">{name}</span>
        <button className="bg-[#050538] text-white px-4 py-1 rounded-md">Relatório</button>
      </div>
    )
  }
  
  export default ProjectElements
  