import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import '../styles/SideBar.css'
import {
  FileText,
  Gamepad2,
  Building,
  Camera,
  Upload,
  Images,
  FileCheck,
  LogOut
} from 'lucide-react'
import logoImage from '../assets/logo-fundo-branco.svg'

const logoutUser = async () => {
  console.log('logoutUser')
  window.location.href = '/'
}

const SideBarProjetos = ({ projetoAtivo }) => {
  const location = useLocation()

  const buildProjectLink = (path) => `${path}?projeto=${projetoAtivo}`;

  return (
    <aside className="sidebar">
      <Link to="/projetos">
        <img id="logo" src={logoImage} alt="Logo Athena" />
      </Link>

      {projetoAtivo && (
        <div className="text-white font-bold text-center mt-4">
          <p className="text-xs">PROJETO</p>
          <p className="text-lg">{projetoAtivo.toUpperCase()}</p>
        </div>
      )}

      <ul>
        <li className={location.pathname === '/projetos' ? 'active' : ''}>
          <Link to="/projetos">
            <FileText />
            <span className="linkText">Projetos</span>
          </Link>
        </li>
        <li className={location.pathname === '/edificios' ? 'active' : ''}>
          <Link to={buildProjectLink("/edificios")}> 
            <Building />
            <span className="linkText">Edifícios</span>
          </Link>
        </li>
        <li className={location.pathname === '/imagens-drone' ? 'active' : ''}>
          <Link to={buildProjectLink("/imagens-drone")}>
            <Camera />
            <span className="linkText">Imagens Drone</span>
          </Link>
        </li>
        <li className={location.pathname === '/upload-imagens' ? 'active' : ''}>
          <Link to={buildProjectLink("/upload-imagens")}>
            <Upload />
            <span className="linkText">Upload de Imagens</span>
          </Link>
        </li>
        <li className={location.pathname === '/analisar-imagens' ? 'active' : ''}>
          <Link to={buildProjectLink("/analisar-imagens")}>
            <Images />
            <span className="linkText">Analisar Imagens</span>
          </Link>
        </li>
        <li className={location.pathname === '/relatorio' ? 'active' : ''}>
          <Link to={buildProjectLink("/relatorio")}>
            <FileCheck />
            <span className="linkText">Relatório</span>
          </Link>
        </li>
      </ul>
      <div id="logoutContainer">
        <button onClick={logoutUser}>
          <LogOut />
          <span className="linkText">Sair</span>
        </button>
      </div>
    </aside>
  )
}

export default SideBarProjetos
