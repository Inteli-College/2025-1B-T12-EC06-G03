// ProjetoLayout.jsx
import React from 'react';
import { Outlet, useSearchParams } from 'react-router-dom';
import SideBarProjetos from './components/SideBarProjetos';

const ProjetoLayout = () => {
  const [params] = useSearchParams();
  const projetoAtivo = params.get("projeto"); // extrai ?projeto=meta

  return (
    <div className="flex">
      <SideBarProjetos projetoAtivo={projetoAtivo} />
      <div className="flex-1">
        <Outlet />
      </div>
    </div>
  );
};

export default ProjetoLayout;
