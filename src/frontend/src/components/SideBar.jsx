import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import '../styles/SideBar.css'
import { Upload, Camera, History, FileCheck, LogOut, Images, FileText, Gamepad2 } from 'lucide-react'
import logoImage from '../assets/logo-fundo-branco.svg'

const logoutUser = async () => {
  console.log('logoutUser')
  window.location.href = '/'
}

const Sidebar = () => {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <img id="logo" src={logoImage} alt="Logo Athena" />
      <ul>
        <li className={location.pathname === '/projetos' ? 'active' : ''}>
          <Link to="/projetos">
            <FileText id="historyIcon" />
            <span className="linkText">Projetos</span>
          </Link>
        </li>
        <li className={location.pathname === '/upload-imagens' ? 'active' : ''}>
          <Link to="/upload-imagens">
            <Upload id="uploadIcon" />
            <span className="linkText">Upload de Imagem</span>
          </Link>
        </li>
        <li className={location.pathname === '/controle-drone' ? 'active' : ''}>
          <Link to="/controle-drone">
            <Gamepad2 id="gamepadIcon" />
            <span className="linkText">Controle Drone</span>
          </Link>
        </li>
        <li className={location.pathname === '/imagens-drone' ? 'active' : ''}>
          <Link to="/imagens-drone">
            <Camera id="cameraIcon" />
            <span className="linkText">Imagens Drone</span>
          </Link>
        </li>
        <li className={location.pathname === '/analisar-imagens' ? 'active' : ''}>
          <Link to="/analisar-imagens">
            <Images id="fileTextIcon" />
            <span className="linkText">Analisar Imagens</span>
          </Link>
        </li>
        <li className={location.pathname === '/relatorio' ? 'active' : ''}>
          <Link to="/relatorio">
            <FileCheck id="fileCheckIcon" />
            <span className="linkText">Relatório</span>
          </Link>
        </li>
        <li className={location.pathname === '/historico' ? 'active' : ''}>
          <Link to="/historico">
            <History id="historyIcon" />
            <span className="linkText">Histórico</span>
          </Link>
        </li>
      </ul>
      <div id="logoutContainer">
        <button onClick={logoutUser}>
          <LogOut id="logoutIcon" />
          <span className="linkText">Sair</span>
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
