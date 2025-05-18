---
title: Descrição do Projeto
sidebar_position: 4
---

import useBaseUrl from '@docusaurus/useBaseUrl';

## Tela de Descrição de Projeto

&emsp; A tela de *Descrição de Projeto* visa descrever as características de um projeto cadastrado na plataforma através de formulários. A tela apresenta as características vinculadas ao projeto e apresenta a opção de edição do projeto.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Frontend página Descrição de Projetos</strong></p>
  <img 
    src={useBaseUrl('/img/pagina_descricao.png')} 
    alt="Frontend página Descrição de Projetos" 
    title="Frontend página Descrição de Projetos" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;Na lateral esquerda, observa-se uma *sidebar* de ações, com botões de redirecionamento para outras páginas do sistema WEB. No corpo da página, encontram-se as informações vinculadas ao projeto, apresentando informações como *Nome, Responsáveis, Empresa Vinculada, Edifícios, Descrição e Logs de Alteração.*

&emsp; Adicionalmente, a tela ainda fornece a funcionalidade de edição do projeto, representado pelo ícone de lápis, ao lado do título principal. Ao acessar o ícone, o usuário é direcionado à tela de edição de projeto, como apresentado na Figura 2.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Edição de Projeto</strong></p>
  <img 
    src={useBaseUrl('/img/pagina_descricao_edicao.png')} 
    alt="Edição de Projeto" 
    title="Edição de Projeto" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### Edição de Projeto

&emsp; Para edição de projeto, o usuário tem a opção de adicionar ou excluir responsáveis pelo projeto - exclui-se um usuário clicando no ícone de *"X"* ao lado do nome do responsável, e adiciona-se um usuário clicando no botão *"Adicionar Responsável"*. Ao interagir com o botão,um campo de texto é liberado para o usuário identificar o novo responśavel.

&emsp; Similarmente à edição de responsáveis, é possível adicionar ou excluir edifícios vinculados ao projeto. Na figura 3, apresenta-se o formulário de inclusão de edifício com as informações necessárias.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 3 - Edição de Edifício</strong></p>
  <img 
    src={useBaseUrl('/img/form_add_edif.png')} 
    alt="Edição de Edifício" 
    title="Edição de Edifício" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### Logs de edição

&emsp; Para audição das alterações realizadas dentro de um projeto, é possível consultar os logs de alterações realizadas, afim de manter integridade e segurança dos relatórios.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 4 - Logs de Alterações</strong></p>
  <img 
    src={useBaseUrl('/img/logs.png')} 
    alt="Logs de Alterações" 
    title="Logs de Alterações" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### Tela de Descrição de Projetos e User Stories

&emsp; Se alinhando especialmente com a necessidade atribuída à persona Maria Silva, a tela **Projetos** entrega acesso aos projetos cadastrados na plataforma, dando acesso às informações principais do projeto, além de permitir edição e atualização dos registros já incluídos. A página ainda fornece os logs de edição de um projeto, necessários para integridade dos dados em futuras auditorias.

&emsp; A interface desenvolvida mantém fidelidade com o Protótipo de Alta Fidelidade, garantindo que a experiência de usuário planejada seja efetivamente entregue no produto final.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 5 - Protótipo de Alta Fidelidade</strong></p>
  <img 
    src={useBaseUrl('/img/descricao_alta_fidelidade.png')} 
    alt="Protótipo de Alta Fidelidade" 
    title="Protótipo de Alta Fidelidade" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>