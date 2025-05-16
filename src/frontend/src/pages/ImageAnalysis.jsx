"use client"

import { useState } from "react"
import ImageCard from "../components/ProjectImageCard"
import placeholder from '../assets/placeholder-icon.svg'

export default function ImageAnalysis() {
  const [images] = useState([
    { id: 1, type: "Retração", imageUrl: {placeholder} },
    { id: 2, type: "Térmica", imageUrl: {placeholder} },
    { id: 3, type: "Térmica", imageUrl: {placeholder} },
    { id: 4, type: "Térmica", imageUrl: {placeholder} },
    { id: 5, type: "Retração", imageUrl: {placeholder} },
    { id: 6, type: "Retração", imageUrl: {placeholder} },
  ])

  const handleViewImage = (id) => {
    alert(`Visualizando imagem ${id}`)
  }

  return (
    <main className="container mx-auto p-6">
      <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {images.map((image) => (
          <ImageCard key={image.id} id={image.id} type={image.type} onView={handleViewImage} />
        ))}
      </div>
    </main>
  )
}
