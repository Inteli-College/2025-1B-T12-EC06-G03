import React from 'react';

const ProjectCard = ({ title, photo, onButtonClick }) => {
    return (
        <div className="border bg- rounded-lg p-4 text-center max-w-xs shadow-md">
            <h3 className="my-4 text-xl font-semibold text-gray-800">{title}</h3>
            <img
                src={photo}
                alt={title}
                className="w-full h-auto rounded-md mb-4"
            />
            <button
                className="px-4 py-2 text-white bg-blue-500 rounded hover:bg-blue-600"
                onClick={onButtonClick}
            >
                Learn More
            </button>
        </div>
    );
};

export default ProjectCard;