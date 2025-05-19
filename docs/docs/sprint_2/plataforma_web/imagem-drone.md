---
title: Imagens do Drone
sidebar_position: 8
---

import useBaseUrl from '@docusaurus/useBaseUrl';

### Tela Imagem do Drone

&emsp; A tela **Imagem do Drone** é responsável por exibir a imagem atual exibida pelo drone, além das que foram capturadas anteriormente e estão associadas ao projeto selecionado. Ela permite a visualização central da imagem em tamanho ampliado e mostra abaixo as miniaturas das imagens capturadas anteriormente. É uma tela essencial para que o operador valide visualmente as coletas realizadas em campo e para que consiga controlá-lo remotamente.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Tela Imagem do Drone (exibição principal)</strong></p>
  <img 
    src={useBaseUrl('/img/tela-imagem-drone2.png')} 
    alt="Imagem principal do drone" 
    title="Imagem principal do drone" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp; Logo ao acessar a página, caso nenhum projeto esteja selecionado, um **modal obrigatório** solicita a escolha do projeto ao qual as imagens serão vinculadas. Essa ação é essencial para garantir a correta organização das imagens capturadas na base de dados do sistema e pode ser vizualizada abaixo:

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Modal de Seleção de Projeto</strong></p>
  <img 
    src={useBaseUrl('/img/modal-selecao-projeto.png')} 
    alt="Modal obrigatório de seleção de projeto" 
    title="Modal obrigatório de seleção de projeto" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp; Caso o usuário altere o projeto após a seleção inicial, um segundo **modal de confirmação** é exibido, evitando trocas acidentais e reforçando a segurança no fluxo de trabalho.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 3 - Modal de Confirmação de Projeto</strong></p>
  <img 
    src={useBaseUrl('/img/modal-confirmacao-projeto.png')} 
    alt="Modal de confirmação ao alterar projeto" 
    title="Modal de confirmação ao alterar projeto" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp; Abaixo da visualização principal, uma seção intitulada **Imagens Capturadas** apresenta as miniaturas das últimas imagens geradas no projeto selecionado. Essa galeria é apresentada em forma de grade com até quatro imagens por linha, facilitando a navegação e revisão do material.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 4 - Tela com galeria de imagens capturadas</strong></p>
  <img 
    src={useBaseUrl('/img/tela-imagem-drone1.png')} 
    alt="Imagens capturadas pelo drone" 
    title="Imagens capturadas pelo drone" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp; A tela também respeita os critérios de usabilidade estabelecidos nas *User Stories*, principalmente no que tange à clareza da interface, à confirmação de ações sensíveis e à organização lógica por projeto. Sua implementação garante ao usuário a visualização rápida do progresso das capturas, sendo muito útil em operações de campo.
