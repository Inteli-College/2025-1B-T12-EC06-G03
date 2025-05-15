import { useState } from 'react';
import ImagensCarregadas from '../components/ImagensCarregadas.jsx';
import placeholder from '../assets/placeholder-icon.svg';

const UploadImagens = () => {
    const [imagens, setImagens] = useState([
        { id: 1, src: placeholder, name: 'Placeholder 1' },
        { id: 2, src: placeholder, name: 'Placeholder 2' },
        { id: 3, src: placeholder, name: 'Placeholder 3' },
    ]);

    const handleImageUpload = (event) => {
        const file = event.target.files[0]; 
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                setImagens((prevImagens) => [
                    ...prevImagens,
                    { id: prevImagens.length + 1, src: e.target.result, name: file.name },
                ]);
            };
            reader.readAsDataURL(file); 
        }
    };

    return (
        <div className="min-h-screen bg-slate-100 p-8">
            <h1 className="text-3xl font-bold mb-6 text-dark-blue">Upload de Imagem</h1>
            <div className="bg-gray-light h-72 flex items-center justify-center rounded-md mb-10">
                <label className="bg-dark-blue text-gray-light px-6 py-2 rounded-xl shadow-md cursor-pointer">
                    Carregar Imagem
                    <input
                        type="file"
                        accept="image/*"
                        className="hidden" 
                        onChange={handleImageUpload} 
                    />
                </label>
            </div>

            <h2 className="text-3xl font-bold mb-6 text-dark-blue">Imagens Carregadas</h2>

            <div className="grid grid-cols-3 gap-x-8 gap-y-8">
                {imagens.map((imagem) => (
                    <ImagensCarregadas 
                        key={imagem.id} 
                        imgSrc={imagem.src} 
                        name={imagem.name} 
                    />
                ))}
            </div>
        </div>
    );
};

export default UploadImagens;