import React from 'react';
import ImagensCarregadas from '../components/ImagensCarregadas.jsx';

const UploadImagens = () => {
    return (
        <div className='min-h-screen bg-slate-100 p-8' >
            <h1 className='text-3xl font-bold mb-6 text-dark-blue'>Upload de Imagem</h1>
            <div className='bg-gray-light h-72 flex items-center justify-center rounded-md mb-10'>
                <button className='bg-dark-blue text-gray-light px-6 py-2 rounded-xl shadow-md'>
                    Carregar Imagem
                </button>
            </div>

            <h2 className='text-3xl font-bold mb-6 text-dark-blue'>Imagens Carregadas</h2>

            <div className='grid grid-cols-3 gap-x-8 gap-y-8'>
                {Array.from({ length: 6 }).map((_, index) => (
                    <ImagensCarregadas key={index} />
                ))}
            </div>
        </div>
    );
};

export default UploadImagens;