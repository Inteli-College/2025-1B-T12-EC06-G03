import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import '../styles/SideBar.css'
import { FileText, Gamepad2, LogOut, Users } from 'lucide-react'
import logoImage from '../assets/logo-fundo-branco.svg'

const logoutUser = async () => {
  console.log('logoutUser')
  window.location.href = '/'
}

const Sidebar = () => {
  const location = useLocation()

  return (
    <aside className="sidebar">
      <Link to="/projetos">
        <img id="logo" src={logoImage} alt="Logo Athena" />
      </Link>
      <ul>
        <li className={location.pathname === '/projetos' ? 'active' : ''}>
          <Link to="/projetos">
            <FileText id="historyIcon" />
            <span className="linkText">Projetos</span>
          </Link>
        </li>
        <li className={location.pathname === '/controle-drone' ? 'active' : ''}>
          <Link to="/controle-drone">
            <Gamepad2 id="gamepadIcon" />
            <span className="linkText">Controle Drone</span>
          </Link>
        </li>
        <li className={location.pathname === '/clientes' ? 'active' : ''}>
          <Link to="/clientes">
            <Users id="usersIcon" />
            <span className="linkText">Clientes</span>
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
