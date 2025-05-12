import { BrowserRouter, Routes, Route } from "react-router-dom"
import Layout from "./Layout.jsx"
import NotFound from "./pages/NotFound"
import LoginPage from "./pages/Login.jsx"
import RecoverPassword from "./pages/RecoverPassword.jsx"

const Router = () => (
  <BrowserRouter>
    <Routes>
      {/* Rotas públicas sem sidebar */}
      <Route path="/" element={<LoginPage />} />
      <Route path="/recover-password" element={<RecoverPassword />} />

      {/* Rotas que terão o layout com sidebar */}
      <Route element={<Layout />}>
        {/* <Route path="upload-imagem" element={<UploadImagem />} />
        <Route path="imagens-drone" element={<ImagensDrone />} />
        <Route path="analisar-imagens" element={<AnalisarImagens />} />
        <Route path="relatorio" element={<Relatorio />} /> */}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  </BrowserRouter>
)

export default Router
