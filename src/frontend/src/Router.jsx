import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";
import ProjectPage from "./pages/Projects.jsx";
import RecoverPassword from "./pages/RecoverPassword.jsx";
import DroneImages from "./pages/DroneImages.jsx";
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
        <Route path="/projetos" element={<ProjectPage />} />
        <Route path="/imagens-drone" element={<DroneImages />} />
        <Route path="/controle-drone" element={<ControleDrone />} />

      </Route>
    </Routes>
  </BrowserRouter>
)

export default Router
