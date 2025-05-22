import React, { useState } from "react";
import logo from "../assets/logo.svg";

const CadastroPage = () => {
  const [nome, setNome] = useState("");
  const [email, setEmail] = useState("");
  const [senha, setSenha] = useState("");
  const [cargo, setCargo] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    // Aqui você pode enviar os dados para o backend
    console.log({ nome, email, senha, cargo });
    // Redirecionar para login após cadastro
    window.location.href = "/login";
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-50">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <div className="flex justify-center mb-8">
          <img src={logo} alt="Logo" className="w-32 h-32" />
        </div>
        <h1 className="text-4xl font-semibold text-center text-gray-800 mb-6">
          Cadastro
        </h1>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="nome" className="block text-sm font-medium text-gray-700">
              Nome
            </label>
            <input
              type="text"
              id="nome"
              value={nome}
              onChange={(e) => setNome(e.target.value)}
              required
              className="w-full px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Digite seu nome completo"
            />
          </div>

          <div className="mb-4">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Digite seu email"
            />
          </div>

          <div className="mb-4">
            <label htmlFor="senha" className="block text-sm font-medium text-gray-700">
              Senha
            </label>
            <input
              type="password"
              id="senha"
              value={senha}
              onChange={(e) => setSenha(e.target.value)}
              required
              className="w-full px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Crie uma senha"
            />
          </div>

          <div className="mb-6">
            <label htmlFor="cargo" className="block text-sm font-medium text-gray-700">
              Cargo
            </label>
            <input
              type="text"
              id="cargo"
              value={cargo}
              onChange={(e) => setCargo(e.target.value)}
              required
              className="w-full px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Informe seu cargo (ex: Engenheiro, Gestor, etc.)"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-dark-blue text-white py-2 rounded-lg hover:bg-blue-darker transition"
          >
            Cadastrar
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            Já tem uma conta?{" "}
            <a href="/login" className="text-blue-500 hover:underline font-medium">
              Faça login
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default CadastroPage;
