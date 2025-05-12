import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import '../styles/SideBar.css'
import { Upload, Camera, FileText, FileCheck, LogOut } from 'lucide-react'
import logoImage from '../assets/logo-fundo-branco.svg'

const logoutUser = async () => {
  console.log('logoutUser')
  window.location.href = '/'
}

const Sidebar = () => {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <img id="logo" src={logoImage} alt="Logo da Vault" />
      <ul>
        <li className={location.pathname === '/upload-imagem' ? 'active' : ''}>
          <Link to="/upload-imagem">
            <Upload id="uploadIcon" />
            <span className="linkText">Upload de Imagem</span>
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
            <FileText id="fileTextIcon" />
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
            <FileText id="historyIcon" />
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
