import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import ProjetoLayout from "./ProjetoLayout.jsx"; // novo layout
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";
import CadastroPage from "./pages/Cadastro.jsx";
import ProjectPage from "./pages/Projects.jsx";
import RecoverPassword from "./pages/RecoverPassword.jsx";
import Relatorios from "./pages/Report.jsx";
import ImageAnalysis from "./pages/ImageAnalysis.jsx";
import DroneImages from "./pages/DroneImages.jsx";
import VisualizarProjeto from "./pages/VisualizarProjeto.jsx";
import ControleDrone from "./pages/ControleDrone.jsx";
import UploadImagens from "./pages/UploadImagens.jsx";
import Clientes from "./pages/Clientes.jsx";
import Edificios from "./pages/Edificios.jsx";

const Router = () => (
  <BrowserRouter>
    <Routes>
      {/* Rotas públicas sem sidebar */}
      <Route path="/" element={<LoginPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/cadastro" element={<CadastroPage />} />
      <Route path="/recover-password" element={<RecoverPassword />} />
      <Route path="*" element={<NotFound />} />
      <Route path = "/visualizar-projeto" element={<VisualizarProjeto />} />

      {/* Rotas com sidebar padrão */}
      <Route element={<Layout />}>
        <Route path="/clientes" element={<Clientes />} />
        <Route path="/projetos" element={<ProjectPage />} />
        <Route path="/controle-drone" element={<ControleDrone />} />
      </Route>

      {/* Rotas com sidebar de projeto */}
      <Route element={<ProjetoLayout />}>
        <Route path="/projeto" element={<VisualizarProjeto />} />
        <Route path="/relatorio" element={<Relatorios />} />
        <Route path="/edificios" element={<Edificios />} />
        <Route path="/analisar-imagens" element={<ImageAnalysis />} />
        <Route path="/imagens-drone" element={<DroneImages />} />
        <Route path="/upload-imagens" element={<UploadImagens />} />
        
      </Route>
    </Routes>
  </BrowserRouter>
);

export default Router;
