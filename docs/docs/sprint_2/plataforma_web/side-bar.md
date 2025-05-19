---
title: Sidebar
sidebar_position: 7
---

import useBaseUrl from '@docusaurus/useBaseUrl';

### Sidebar de Navegação

&emsp; O componente de **sidebar** é o elemento responsável pela navegação principal do sistema WEB. Posicionado à esquerda da interface, ele está disponível em todas as telas após o login, garantindo acesso rápido às funcionalidades centrais da aplicação. A sidebar se apresenta como um menu vertical fixo, que utiliza ícones representativos e texto (em versão expandida) para facilitar o entendimento das opções.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Sidebar Compacta</strong></p>
  <img 
    src={useBaseUrl('/img/sidebar-retraida.png')} 
    alt="Sidebar Compacta" 
    title="Sidebar Compacta" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;Na versão compacta (Figura 1), a sidebar exibe apenas os ícones, otimizando espaço e mantendo a usabilidade em telas menores ou quando o usuário prefere uma interface mais limpa. Ao passar o cursor e expandir a sidebar, os textos descritivos de cada item são exibidos, conforme ilustrado na Figura 2.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Sidebar Expandida</strong></p>
  <img 
    src={useBaseUrl('/img/sidebar-expandida.png')} 
    alt="Sidebar Expandida" 
    title="Sidebar Expandida" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;A sidebar contém as seguintes opções de navegação, organizadas verticalmente:

- **Projetos**: Tela inicial pós-login, exibe a lista de projetos e suas imagens associadas.
- **Upload de Imagem**: Permite o envio de novas imagens para análise.
- **Controle Drone**: Redireciona para a interface de operação e monitoramento do drone.
- **Imagens Drone**: Galeria com imagens capturadas diretamente pelos drones.
- **Analisar Imagens**: Tela dedicada à execução dos algoritmos de detecção e classificação de fissuras.
- **Relatório**: Geração e visualização dos relatórios técnicos de cada projeto.
- **Histórico**: Histórico de atividades e projetos acessados pelo usuário.
- **Sair**: Encerra a sessão atual do sistema.

&emsp;A experiência do usuário foi projetada para garantir fluidez entre páginas e consistência visual com os demais componentes da interface. A utilização de ícones intuitivos proporciona fácil associação de funcionalidades mesmo sem o uso de texto explicativo.

&emsp;O componente de sidebar também respeita as *User Stories* mapeadas, especialmente no que tange à necessidade de acessibilidade e navegação eficiente por parte de usuários não técnicos, como a persona **Maria Silva**, garantindo que os fluxos operacionais do sistema sejam simples e rápidos.
