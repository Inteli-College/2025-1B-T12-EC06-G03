---
title: Atualizações no Front-End
sidebar_position: 1
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Atualizações no Front-End

#### Atualizações no Front-End

## Introdução

&emsp;Durante o ciclo mais recente de desenvolvimento, foram realizadas **alterações significativas** na arquitetura e experiência de usuário do front-end da aplicação. Essas mudanças responderam a dois principais objetivos:

1. **Organizar a navegação por projeto**, com componentes e telas que respondem dinamicamente ao contexto do projeto aberto.
2. **Aprimorar a interface para ações críticas**, como análise de imagens, visualização de relatórios e gerenciamento de dados associados (clientes, edifícios, responsáveis, etc).

&emsp;A seguir, detalhamos o que foi alterado, os motivos por trás dessas decisões e como essas mudanças impactam positivamente a evolução do projeto.


## Alterações Gerais e Justificativas

### 1. **Sidebar contextual por projeto**

&emsp;Criamos um componente `SidebarProjetos` que aparece apenas quando um projeto está aberto. Isso **evita poluição visual nas telas principais** e garante que o usuário esteja sempre navegando pelas opções corretas do projeto em questão, como “Imagens Drone”, “Relatórios” e “Análise de Imagens”.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Atualização na SideBar</strong></p>
  <img 
    src={useBaseUrl('/img/atualizacao1.png')} 
    alt="Atualização na SideBar" 
    title="Atualização na SideBar" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### 2. **Parametrização via URL**

&emsp;Telas como `/relatorio`, `/analise` e `/drone` agora aceitam parâmetros como `?projeto=usp`. Isso permite que os dados de cada projeto sejam **carregados dinamicamente**, melhorando a escalabilidade e evitando redundância de rotas.

