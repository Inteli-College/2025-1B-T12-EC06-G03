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

### 3. **Estado do Projeto (Em Andamento / Finalizado)**

&emsp;Adicionou-se **um status visível de andamento** nos cards dos projetos, bem como dentro do relatório. Um botão “Encerrar Projeto” também foi incluído com um modal de confirmação seguro. Isso reforça o **controle de fluxo e ciclo de vida do projeto**, além de evitar alterações acidentais.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Atualização Status do Projeto</strong></p>
  <img 
    src={useBaseUrl('/img/atualizacao2.png')} 
    alt="Atualização Status do Projeto" 
    title="Atualização Status do Projeto" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### 4. **Substituição de `confirm()` por Modal**

&emsp;Removemos o uso do `confirm()` por boas práticas de lint e segurança. Em seu lugar, foi desenvolvido um **modal de confirmação customizado**, com estilização em Tailwind e comportamento mais fluido. Isso melhora a **coerência visual e acessibilidade**.

### 5. **Proteção de logs no relatório**

&emsp;Os campos de **logs de alteração foram excluídos do modo de edição**, garantindo que edições de relatório não comprometam o histórico de ações. Isso é essencial para **auditoria, rastreabilidade e segurança da informação**.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 3 - Logs de Edição do Relatório</strong></p>
  <img 
    src={useBaseUrl('/img/atualizacao3.png')} 
    alt="Logs de Edição do Relatório" 
    title="Logs de Edição do Relatório" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

### 6. **Tratamento de erros na tela de relatórios**

&emsp;Adicionamos tratamento para evitar quebra de tela quando um projeto selecionado não possui dados completos (`porcentagemFissuras`, por exemplo). Isso corrige um erro crítico de runtime e **melhora a resiliência da aplicação**.

### 7. **Padronização visual e responsiva**

&emsp;Foram aplicadas diversas correções no layout utilizando Tailwind para garantir que todos os componentes sigam uma **identidade visual consistente**, com melhor experiência para o usuário final e reforço ao branding.

### 8. **Gestão do ciclo de análise de imagens**

&emsp;Agora, as imagens capturadas por drones ou carregadas manualmente apresentam **status visível indicando se já foram processadas pelo modelo**. Após o processamento, o sistema permite que **um funcionário aprove ou não o resultado da análise**. Apenas imagens aprovadas passam a constar no relatório.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 4 - Status das Imagens para o Modelo</strong></p>
  <img 
    src={useBaseUrl('/img/atualizacao4.png')} 
    alt="Status das Imagens para o Modelo" 
    title="Status das Imagens para o Modelo" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;Essa funcionalidade fortalece a **validação humana** antes da tomada de decisão, garantindo mais **confiabilidade e precisão na documentação técnica**.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 5 - Aprovar Imagens Retonadas do Modelo</strong></p>
  <img 
    src={useBaseUrl('/img/atualizacao5.png')} 
    alt="Aprovar Imagens Retonadas do Modelo" 
    title="Aprovar Imagens Retonadas do Modelo" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

## Telas Novas Criadas

&emsp;Além disso, ao longo do processo, foi vista a necessidade da criação de novas telas. Assim, essas telas estruturam o sistema em módulos claros e **viabilizam escalabilidade futura** — como permissões de acesso, histórico por edifício e dashboards de cliente.

| Tela                | Finalidade                                                                |
| ------------------- | ------------------------------------------------------------------------- |
| **Cadastro**        | Permitir o registro de novos usuários ou entidades relevantes ao projeto. |
| **Clientes**        | Visualizar e gerenciar dados dos clientes associados a cada projeto.      |
| **Edifícios**       | Cadastro e visualização de prédios vinculados a cada projeto.             |
| **ProjectPage**     | Página de entrada individual por projeto, que gerencia contexto e rotas.  |
| **SidebarProjetos** | Navegação lateral exibida apenas ao abrir um projeto específico.          |


## Conclusão

&emsp;As mudanças realizadas nesta Sprint representam um **avanço estratégico em organização, usabilidade e escalabilidade** do sistema. Ao reorganizar a navegação por projeto, separar responsabilidades em telas específicas e garantir a integridade das ações críticas (como edição de relatório, status de projeto e aprovação de imagens), o projeto se torna mais:

* **Clareado para o usuário final**, com informações acessíveis e centralizadas.
* **Manutenível e escalável**, facilitando o onboarding de novos devs e expansão futura.
* **Confiável e robusto**, com validações de etapa, menos pontos de falha e mais previsibilidade nas ações.

&emsp;Com isso, o sistema se posiciona de forma sólida para evoluir em direção a integrações mais avançadas, como painéis analíticos, permissões por papel e auditoria contínua das imagens e estruturas.


