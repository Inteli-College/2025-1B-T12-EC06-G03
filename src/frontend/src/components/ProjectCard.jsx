import placeholder from '../assets/placeholder-icon.svg'

const ProjectCard = ({ name }) => (
  <div className="bg-blue-300 rounded-md overflow-hidden shadow-md">
    <div className="bg-gray-100 m-4 h-36 rounded-md flex items-center justify-center p-6">
      <img src={placeholder} alt="Placeholder" className="max-h-full max-w-full object-contain" />
    </div>
    <div className="p-4 flex justify-between items-center">
      <span className="font-medium text-lg">{name}</span>
      <button className="bg-dark-blue text-white px-4 py-2 rounded-md hover:bg-blue-darker">Relatório</button>
    </div>
  </div>
)

export default ProjectCard
