import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout.jsx";
import NotFound from "./pages/NotFound.jsx";


const Router = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Rota para não encontrado */}
        <Route path="*" element={<NotFound />} />

        {/* Rotas sem sidebar */}

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
