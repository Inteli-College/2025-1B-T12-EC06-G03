import React, { useState } from "react";
import logo from "../assets/logo.svg";
import httpClient from "../httpClient";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [senha, setSenha] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const response = await httpClient.post("/auth/login", { email, senha });
      const { token } = response.data;
      localStorage.setItem("token", token);
      window.location.href = "/projetos";
    } catch (err) {
      setError("Email ou senha inválidos");
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-50">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <div className="flex justify-center mb-8">
          <img src={logo} alt="Logo" className="w-32 h-32" />
        </div>
        <h1 className="text-4xl font-semibold text-center text-gray-800 mb-6">Login</h1>

        {error && <p className="text-red-500 mb-4">{error}</p>}

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              id="email"
              type="email"
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
              id="senha"
              type="password"
              value={senha}
              onChange={(e) => setSenha(e.target.value)}
              required
              className="w-full px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Digite sua senha"
            />
          </div>

          <div className="flex justify-between items-center mb-6">
            <a href="/recover-password" className="text-sm text-blue-500 hover:underline">
              Esqueci a senha
            </a>
          </div>

          <button
            type="submit"
            className="w-full bg-dark-blue text-white py-2 rounded-lg hover:bg-blue-darker transition"
          >
            Entrar
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            Não tem uma conta?{" "}
            <a href="/cadastro" className="text-blue-500 hover:underline font-medium">
              Cadastre-se
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
