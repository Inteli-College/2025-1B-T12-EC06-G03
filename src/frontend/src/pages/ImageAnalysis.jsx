import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import ImageCard from "../components/ProjectImageCard";
import placeholder from "../assets/placeholder-icon.svg";

export default function ImageAnalysis() {
  const [searchParams] = useSearchParams();
  const projetoAtivo = searchParams.get("projeto");

  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    const fetchFissuras = async () => {
      try {
        const response = await fetch("http://localhost:8080/api/fissuras");
        if (!response.ok) throw new Error("Erro ao buscar fissuras");
        const fissuras = await response.json();

        const imagensDoProjeto = fissuras.filter(f => f.imagem?.projeto?.nome?.toLowerCase() === projetoAtivo?.toLowerCase());

        const formatadas = imagensDoProjeto.map(f => ({
          id: f.imagem.id,
          caminho: `https://efinfalxxeaqfkvboewx.supabase.co/storage/v1/object/public/img-projects/${f.imagem.caminhoArquivo}`,
          label: f.tipo,
          bbox: f.coordenadas,
          confidence: f.confianca,
          aprovado: f.aprovado || false,
          aprovadoPor: f.aprovadoPor || null,
        }));

        setImages(formatadas);
      } catch (err) {
        console.error("Erro ao carregar imagens analisadas:", err);
      }
    };

    fetchFissuras();
  }, [projetoAtivo]);

  const handleViewImage = (image) => {
    setSelectedImage(image);
  };

  const handleAprovar = async () => {
    const nomeUsuario = "Especialista 1";
    setImages((prev) =>
      prev.map((img) =>
        img.id === selectedImage.id
          ? { ...img, aprovado: true, aprovadoPor: nomeUsuario }
          : img
      )
    );
    setSelectedImage(null);

    await fetch(`http://localhost:8080/api/fissuras/${selectedImage.id}/aprovar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aprovado: true, aprovadoPor: nomeUsuario })
    });
  };

  return (
    <main className="container mx-auto p-6">
      <h1 className="text-4xl font-bold text-black mb-10">Analisar Imagens</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {images.map((image) => (
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
              type={image.label}
              imageUrl={image.caminho}
              onView={() => handleViewImage(image)}
            />
          </div>
        ))}
      </div>

      {selectedImage && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-4 rounded shadow-lg max-w-md w-full">
            <h2 className="text-lg font-semibold mb-4">
              Imagem #{selectedImage.id} ({selectedImage.label})
            </h2>
            <img
              src={selectedImage.caminho || placeholder}
              alt={`Imagem ${selectedImage.id}`}
              className="max-h-[300px] w-auto mx-auto mb-4 rounded object-contain"
            />

            <p className="text-sm text-gray-700 mb-2">
              <strong>Coordenadas da Fissura:</strong><br />
              x: {selectedImage.bbox?.x}, y: {selectedImage.bbox?.y}, largura: {selectedImage.bbox?.width}, altura: {selectedImage.bbox?.height}
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
