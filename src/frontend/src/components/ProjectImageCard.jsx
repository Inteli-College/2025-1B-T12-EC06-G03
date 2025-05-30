"use client"

const ImageCard = ({ id, type, imageUrl, onView }) => {
  return (
    <div className="bg-blue-300 rounded-md overflow-hidden shadow-md">
      <div className="bg-white m-3 h-36 rounded-md flex items-center justify-center">
        <img 
          src={imageUrl} 
          alt={`Imagem ${id}`} 
          className="object-contain h-full max-h-36 rounded" 
        />
      </div>
      <div className="p-4 flex justify-between items-center">
        <span className="font-medium text-lg">{type}</span>
        <button 
          onClick={() => onView(id)} 
          className="bg-dark-blue hover:bg-blue-darker text-white px-4 py-1 rounded-md transition-colors"
        >
          Ver
        </button>
      </div>
    </div>
  )
}

export default ImageCard
