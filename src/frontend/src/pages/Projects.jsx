"use client";

import { useState } from "react";
import ProjectRecent from '../components/ProjectRecent'
import ProjectAll from '../components/ProjectAll'

export default function ProjectPage() {
    const [searchTerm, setSearchTerm] = useState("");
    const [projects] = useState([
        { id: 1, name: "USP" },
        { id: 2, name: "IBM" },
        { id: 3, name: "Meta" },
        { id: 4, name: "Apontar" },
    ]);

    const recentProjects = projects.slice(0, 4);
    const filteredProjects = projects.filter((project) =>
        project.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleCreateProject = () => {
        alert("Criar novo projeto");
    };

    return (
        <main className="container mx-auto p-6 max-w-">
            <ProjectRecent projects={recentProjects} onCreateProject={handleCreateProject} />

            <ProjectAll
                projects={filteredProjects}
                searchTerm={searchTerm}
                onSearchChange={setSearchTerm}
            />
        </main>
    );
}