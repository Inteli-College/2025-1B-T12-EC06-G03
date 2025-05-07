import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";
import LoginPage from "./pages/Login.jsx";


const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Rota para não encontrado */}
        <Route path="*" element={<NotFound />} />

        {/* Rotas sem sidebar */}
        <Route path="/" element={<LoginPage />} />

        {/* Rotas que terão o layout com sidebar */}
        <Route element={<Layout />}>
          {/* Rotas protegidas
        
          <Route element={<PrivateRoute />}>
          </Route> */}
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default Router;
