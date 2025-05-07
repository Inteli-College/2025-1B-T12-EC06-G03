import React, { useState } from "react";
import logo from "../assets/logo.svg";
const RecoverPassword = () => {
  const [email, setEmail] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Email:", email);
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-50">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-lg">
        <div className="flex justify-center mb-8">
          <img src={logo} alt="Logo" className="w-32 h-32" />
        </div>
        <h1 className="text-3xl font-semibold text-center text-gray-800 mb-6">
        Recuperação de Senha
        </h1>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full mb-10 px-4 py-2 border rounded-lg border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Digite seu email"
            />
          </div>
          <button
            type="submit"
            className="w-full bg-dark-blue text-white py-2 rounded-lg hover:bg-blue-darker transition"
          >
            Entrar
          </button>
        </form>
      </div>
    </div>
  );
};

export default RecoverPassword;
