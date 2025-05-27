import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import ImageCard from "../components/ProjectImageCard";
import placeholder from "../assets/placeholder-icon.svg";

export default function ImageAnalysis() {
  const [searchParams] = useSearchParams();
  const projetoAtivo = searchParams.get("projeto");

  const [images, setImages] = useState([
    {
      id: 1,
      type: "Retração",
      imageUrl: placeholder,
      aprovado: false,
      aprovadoPor: "",
      projeto: "usp",
      bbox: { x: 100, y: 80, width: 200, height: 100 },
      confidence: 0.92,
    },
    {
      id: 2,
      type: "Térmica",
      imageUrl: placeholder,
      aprovado: false,
      aprovadoPor: "",
      projeto: "meta",
      bbox: { x: 120, y: 90, width: 150, height: 120 },
      confidence: 0.87,
    },
    {
      id: 3,
      type: "Térmica",
      imageUrl: placeholder,
      aprovado: true,
      aprovadoPor: "Especialista 1",
      projeto: "usp",
      bbox: { x: 110, y: 70, width: 180, height: 110 },
      confidence: 0.95,
    },
  ]);

  const [selectedImage, setSelectedImage] = useState(null);

  const handleViewImage = (image) => {
    setSelectedImage(image);
  };

  const handleAprovar = () => {
    const nomeUsuario = "Especialista 1";
    setImages((prev) =>
      prev.map((img) =>
        img.id === selectedImage.id
          ? { ...img, aprovado: true, aprovadoPor: nomeUsuario }
          : img
      )
    );
    setSelectedImage(null);
  };

  const imagensFiltradas = images.filter((img) => img.projeto === projetoAtivo);

  return (
    <main className="container mx-auto p-6">
      <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {imagensFiltradas.map((image) => (
          <div key={image.id} className="relative border rounded-lg p-2 bg-white shadow">
            <div className="mb-2">
              {image.aprovado ? (
                <span className="text-green-700 bg-green-100 px-2 py-1 rounded text-sm font-medium">
                  Aprovado {image.aprovadoPor && `por ${image.aprovadoPor}`}
                </span>
              ) : (
                <span className="text-yellow-800 bg-yellow-100 px-2 py-1 rounded text-sm font-medium">
                  Esperando Aprovação
                </span>
              )}
            </div>
            <ImageCard
              id={image.id}
              type={image.type}
              onView={() => handleViewImage(image)}
            />
          </div>
        ))}
      </div>

      {selectedImage && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-6 rounded shadow-lg max-w-lg w-full">
            <h2 className="text-xl font-semibold mb-4">
              Imagem #{selectedImage.id} ({selectedImage.type})
            </h2>
            <img
              src={selectedImage.imageUrl}
              alt={`Imagem ${selectedImage.id}`}
              className="w-full h-auto mb-4 rounded"
            />

            <p className="text-sm text-gray-700 mb-2">
              <strong>Coordenadas da Fissura:</strong><br />
              x: {selectedImage.bbox.x}, y: {selectedImage.bbox.y}, largura: {selectedImage.bbox.width}, altura: {selectedImage.bbox.height}
            </p>

            <p className="text-sm text-gray-700 mb-4">
              <strong>Confiança da Detecção:</strong> {(selectedImage.confidence * 100).toFixed(1)}%
            </p>

            {!selectedImage.aprovado ? (
              <button
                onClick={handleAprovar}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Aprovar Imagem
              </button>
            ) : (
              <p className="text-green-600 font-medium">
                Imagem já aprovada por {selectedImage.aprovadoPor}
              </p>
            )}
            <div className="mt-4 text-right">
              <button
                onClick={() => setSelectedImage(null)}
                className="text-sm text-gray-500 hover:underline"
              >
                Fechar
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
