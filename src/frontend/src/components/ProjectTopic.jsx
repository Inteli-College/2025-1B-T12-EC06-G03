import React from 'react';

const ProjectTopic = ({ title, onButtonClick }) => {
    return (
        <div className='flex flex-col gap-4 bg-blue-dark rounded-2xl p-6 text-center shadow-md' style={{ width: '350px', height: '300px' }}>
            <div className='border-2 rounded-lg bg-white border-black w-full h-2/3'>
            </div>
            <div className='grid grid-cols-2 content-around gap-2'>
                <div>
                    <h3 className="my-4 text-4xl text-left font-semibold text-gray-800">{title}</h3>
                </div>
                <div className='flex contend-end py-4'>
                    <button
                        className="flex items-center justify-center text-lg text-white text-center bg-blue-darker w-full h-full rounded-xl hover:bg-dark-blue"
                        onClick={onButtonClick}
                    > Relatório
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ProjectTopic;