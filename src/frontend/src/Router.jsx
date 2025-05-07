import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";
import RecoverPassword from "./pages/RecoverPassword.jsx";
import VisualizarProjeto from "./pages/VisualizarProjeto.jsx";

const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Rota para não encontrado */}
        <Route path="*" element={<NotFound />} />

        {/* Rotas sem sidebar */}
        <Route path="/" element={<LoginPage />} />
        <Route path="/recover-password" element={<RecoverPassword />} />

        {/* Rotas que terão o layout com sidebar */}
        <Route element={<Layout />}>
          <Route path="/projeto" element={<VisualizarProjeto />} />
          {/* Rotas protegidas
        
          <Route element={<PrivateRoute />}>
          </Route> */}
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default Router;
