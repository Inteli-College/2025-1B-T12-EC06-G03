const ProjectElement = ({ name }) => {
    return (
      <div className="bg-blue-300 rounded-md p-5 flex justify-between items-center shadow-md">
      <span className="font-medium text-lg">{name}</span>
      <button className="bg-[#050538] text-white px-5 py-2 rounded-md">Relatório</button>
      </div>
    )
  }
  
  export default ProjectElement
  