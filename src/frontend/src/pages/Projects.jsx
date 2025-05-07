import React from 'react';
import ProjectCard from '../components/ProjectCard';

const ProjectPage = () => {
    const handleButtonClick = () => {
        alert('Button clicked!');
    };

    return (
        <div className="flex justify-center mt-5">
            <ProjectCard
                title="teste"
                photo="das"
                onButtonClick={handleButtonClick}
            />
        </div>
    );
};

export default ProjectPage;