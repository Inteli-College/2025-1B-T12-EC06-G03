import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";
import ProjectPage from "./pages/Projects.jsx";
import RecoverPassword from "./pages/RecoverPassword.jsx";
import ControleDrone from "./pages/ControleDrone.jsx";

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
        {/* <Route path="upload-imagem" element={<UploadImagem />} /> */}
        <Route path="/controle-drone" element={<ControleDrone />} />
        {/* <Route path="/imagens-drone" element={<ImagensDrone />} /> */}
        {/* <Route path="analisar-imagens" element={<AnalisarImagens />} />
        <Route path="relatorio" element={<Relatorio />} /> */}
      </Route>
    </Routes>
  </BrowserRouter>
)

export default Router
