import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";
import ProjectPage from "./pages/Projects.jsx";
import RecoverPassword from "./pages/RecoverPassword.jsx";
import Relatorios from "./pages/Relatorios.jsx";
import ImageAnalysis from "./pages/ImageAnalysis.jsx";
import DroneImages from "./pages/DroneImages.jsx";
import VisualizarProjeto from "./pages/VisualizarProjeto.jsx";
import ControleDrone from "./pages/ControleDrone.jsx";
import UploadImagens from "./pages/UploadImagens.jsx";

const Router = () => (
  <BrowserRouter>
    <Routes>
      {/* Rotas públicas sem sidebar */}
      <Route path="/" element={<LoginPage />} />
      <Route path="/recover-password" element={<RecoverPassword />} />
      <Route path="*" element={<NotFound />} /> 

        {/* Rotas sem sidebar */}
        <Route path="/" element={<LoginPage />} />
        <Route path="/recover-password" element={<RecoverPassword />} />
      {/* Rotas que terão o layout com sidebar */}
      <Route element={<Layout />}>
        <Route path="/projects" element={<ProjectPage />} />
        <Route path="/relatorios" element={<Relatorios />} />
        <Route path="/analisar-imagens" element={<ImageAnalysis />} />
        {/* <Route path="upload-imagem" element={<UploadImagem />} />
        <Route path="imagens-drone" element={<ImagensDrone />} />
        <Route path="analisar-imagens" element={<AnalisarImagens />} />
        <Route path="relatorio" element={<Relatorio />} /> */}
        <Route path="/projeto" element={<VisualizarProjeto />} />
        <Route path="/projetos" element={<ProjectPage />} />
        <Route path="/imagens-drone" element={<DroneImages />} />
        <Route path="/controle-drone" element={<ControleDrone />} />
        <Route path="/upload-imagens" element={<UploadImagens />} />
      </Route>
    </Routes>
  </BrowserRouter>
)

export default Router
