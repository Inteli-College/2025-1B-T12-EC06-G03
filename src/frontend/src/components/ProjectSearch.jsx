"use client"

import { Search } from "lucide-react"

const ProjectSearch = ({ searchTerm, onSearchChange }) => {
  return (
    <div className="flex justify-end mb-6">
      <div className="relative w-full max-w-md">
        <input
          type="text"
          placeholder="Pesquisar"
          value={searchTerm}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full bg-gray-300 rounded-md py-3 px-5 pr-12 focus:outline-none text-lg"
        />
        <Search className="absolute right-3 top-3.5 text-gray-500" size={24} />
      </div>
    </div>
  )
}

export default ProjectSearch
