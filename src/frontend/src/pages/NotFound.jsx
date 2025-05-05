import React from "react";

const NotFound = () => {
  return (
    <div className="flex items-center justify-center h-screen bg-gray-50">
      <div className="flex flex-col items-center space-y-4">
        <h1 className="text-6xl font-bold">404</h1>
        <p className="text-lg">Oops! Esta página não foi encontrada!</p>
        <button
          onClick={() => window.history.back()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Voltar para a última página acessada
        </button>
      </div>
    </div>
  );
};

export default NotFound;
