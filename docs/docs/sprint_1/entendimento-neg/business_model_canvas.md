---
title: Business Model Canvas
sidebar_position: 1
---

import useBaseUrl from '@docusaurus/useBaseUrl';

## 1. Introdução

&emsp; O Business Model Canvas (BMC) é uma ferramenta estratégica criada por Alexander Osterwalder e Yves Pigneur, cujo objetivo é representar visualmente os principais elementos que compõem um modelo de negócio específico. Estruturado em nove blocos interdependentes, o BMC permite compreender como uma organização cria, entrega e captura valor por meio de suas atividades, recursos, parceiros e canais de relacionamento com o cliente [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp; A simplicidade visual do Canvas favorece sua aplicação prática em contextos diversos, sendo amplamente adotado por startups, empresas consolidadas e projetos de base tecnológica, cenário de aplicação deste documento. Sua estrutura orienta decisões estratégicas e facilita a comunicação entre stakeholders, especialmente em estágios iniciais de desenvolvimento, onde hipóteses sobre o mercado, o produto e os clientes ainda estão sendo validadas. Na figura 1, encontra-se uma representação visual dos blocos de estudo de compreensão dos modelos de negócio estudados pelo Business Model Canvas [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Representação visual Business Model Canvas</strong></p>
  <img 
    src={useBaseUrl('/img/bmc_representation.png')}
    alt="Representação visual Business Model Canvas" 
    title="Representação visual Business Model Canvas" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Business Model Generation (2010)</p>
</div>

&emsp; Portanto, no âmbito das análises de mercado, o BMC atua como um instrumento de identificação de oportunidades e riscos, facilitando o alinhamento entre proposta de valor e segmentos de clientes. Além disso, ao evidenciar fontes de receita, estrutura de custos e parcerias-chave, o modelo contribui para a sustentabilidade e escalabilidade do projeto [(VICELLU & TOLFO, 2016)](#referências-bibliográficas).


## 2. Business Model Canvas aplicado ao projeto Athena

&emsp; Na Figura 2, expõe-se o estudo de Business Model Canvas do projeto Athena.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Business Model Canvas do projeto Athena</strong></p>
  <img 
    src={useBaseUrl('/img/bmc.png')}
    alt="Business Model Canvas do projeto Athena" 
    title="Business Model Canvas do projeto Athena" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaboração própria (2025)</p>
</div>

&emsp; Abaixo, aprofunda-se no estudo e entendimento dos blocos de compreensão do modelo de negócios adotado para o projeto Athena, bem como a justificativa dos tópicos levantados em cada item. 

### 2.1. Proposta de Valor

&emsp;A Proposta de Valor, no Business Model Canvas, refere-se à gama de serviços ou soluções que uma empresa entrega aos seus clientes. Frequentemente, destaca-se a proposta de valor como principal agente responsável pela escolha de uma empresa sob outra por parte do cliente. A Proposta de Valor, portanto, consiste nos serviços fornecidos pela empresa analisada em seguindo os requerimentos do Segmento de Clientes abordado [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp;Além do embasamento teórico, a Proposta de Valor do projeto Athena considera as necessidades levantadas pelas Personas e User Stories definidas no estudo de UX produzido. Portanto, considerando a [User Stories 01 e 02](/docs/docs/sprint_1/ux/user_stories.md), vinculadas à persona Maria Silva, o sistema concentra-se na automatização da detecção e classificação de fissuras, proporcionando maior eficiência no processo de identificação de rachaduras, redução significativa nos custos de inspeções manuais e manutenções, além de contribuir para o processo de manutenções preventivas nas estruturas. Pela perspectiva do técnico que utilizará o sistema Athena, considera-se as [User Stories 01 e 02](/docs/docs/sprint_1/ux/user_stories.md) vinculadas à persona Rafael Lima, que requer a disponibilização de uma interface intuitiva pela plataforma, permitindo o upload de imagens automáticas advindas diretamente do drone, além de upload manual, e a geração de relatórios técnicos. A utilização de bases de dados permitirá o monitoramento da evolução das fissuras ao longo do tempo, agregando ainda mais valor à manutenção preventiva [(TAPI, 2025)](#referências-bibliográficas).

### 2.2. Segmentos de Clientes

&emsp;No contexto do Business Model Canvas, o [Segmento de Clientes](#22-segmentos-de-clientes) diz respeito aos diferentes grupos de pessoas ou organizações que a empresa pretende atingir, de maneira a considerar as diferentes necessidades e comportamentos do público abrangido por suas soluções [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp;Portanto, no projeto Athena, os segmentos de clientes priorizados incluem empresas de engenharia civil, empresas de manutenção predial, consultorias e auditorias técnicas, construtoras focadas em manutenção pós-obra, administradoras de condomínios e órgãos públicos responsáveis pela fiscalização de edificações [(TAPI, 2025)](#referências-bibliográficas).

### 2.3. Canais

&emsp;O bloco de Canais, no contexto BMC, define a maneira em que a empresa se comunica e atinge os grupos definidos no Segmento de Clientes para entrega da Proposta de Valor [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp; Os clientes definidos no [Segmento de Clientes](#22-segmentos-de-clientes) poderão ser alcançados através de vendas consultivas diretas, parcerias estratégicas com fabricantes e operadores de drones, participações em feiras e eventos de inovação e construção civil, além da divulgação em revistas técnicas especializadas.

### 2.4. Relacionamento com Clientes

&emsp; O quadro de Relacionamento com Clientes, como o nome sugere, reflete o tipo de relacionamento que a empresa pretende manter com Segmentos de Clientes específicos. Destacam-se os relacionamentos pessoais e automatizados, geralmente motivados por aquisição e retenção de clientes [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp; O relacionamento com os clientes será construído com base em suporte técnico especializado, disponibilizado via canais digitais (como feedbacks de melhoria e bugs direto da plataforma, e-mails, reuniões virtuais, dentre outros), além da oferta de materiais de tutoria para uso da plataforma. O feedback contínuo dos usuários será incorporado como prática para a evolução da plataforma (considerando usabilidade e tempo de resposta das requisições do usuário, por exemplo) e do algoritmo de classificação das fissuras, como estratégia de retenção dos clientes. A caráter emergencial, disponibiliza-se um canal para recebimento de urgências no uso da plataforma, para suporte ao cliente.

### 2.5. Fontes de Receita

&emsp; No que tange as Fontes de Receita no modelo BMC, sua finalidade é representar a quantidade de caixa gerado a partir de cada um dos [Segmento de Clientes](#22-segmentos-de-clientes), podendo envolver dois tipos de fontes [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas):

1. Receitas a partir de transações únicas pelo cliente;
2. Receitas recorrentes a partir de planos ou suporte ao cliente após a compra do produto/solução.

&emsp; As fontes de receita serão estruturadas principalmente na venda de licenças anuais do software, com upload ilimitado de imagens, no modelo Software as a Service (SaaS), sobre regime de receitas recorrentes a partir de licenças, interessante para o sustento do produto à longo prazo. Adicionalmente, será oferecida uma modalidade de cobrança por upload de imagens processadas via plataforma, sem a necessidade de integração do software à um Veículo Aéreo Não-Tripulado, mas apenas classificando as imagens enviadas pelo cliente. Serviços complementares incluirão treinamentos especializados e consultoria para integração do sistema em rotinas internas de coleta de imagens e avaliação de fissuras, sobre regime de transação única pelo cliente.

### 2.6. Recursos Principais

&emsp; O bloco de análise de Recursos Principais destaca os pontos mais importantes requeridos para o funcionamento do modelo de negócios e entrega de valor ao cliente. É importante destacar que, a depender de quais os produtos e soluções disponibilizadas pela instituição, os Recursos Principais podem ser físicos, financeiros, intelectuais ou humanos [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp; Portanto, entende-se que para viabilizar o projeto, os Recursos Principais necessários se classificam abaixo:

- Físicos: Dispositivos de captura de imagens em alta resolução, Veículos Aéreos Não Tripulados (para identificação de fissuras em altas elevações), tablets e/ou computadores para inicialização e monitoramento do processo de captura de imagens;

- Humanos: Equipe de desenvolvimento especializada em Machine Learning e Algoritmos de Visão Computacional, infraestrutura de armazenamento e processamento de dados e desenvolvimento de plataformas WEB2;

- Intelectuais: Parcerias técnicas com engenheiros civis para validação primordial das classificações feitas pelo algoritmo desenvolvido e uma base de dados já classificada para treinamento algoritmo preditivo.

### 2.7. Atividades-Chave

&emsp; As Atividades-Chaves, no ponto de vista da modelagem BMC, refere-se às ações fundamentais pra operação do modelo de negócios. Portanto, estas ações são requeridas para a [Proposta de Valor](#21-proposta-de-valor), manutenção do [Relacionamento com Clientes](#24-relacionamento-com-clientes) e [recebimento das receitas](#25-fontes-de-receita) [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp; Entre as atividades-chave do projeto, destacam-se o desenvolvimento contínuo do algoritmo de detecção e classificação de fissuras, a validação técnica das fissuras detectadas com apoio de especialistas da área de engenharia civil, a integração da plataforma desenvolvida pelo grupo Athena aos dispositivos de captura de imagens, fornecimento de uma interface gráfica intuitiva para os usuários e a prestação de suporte técnico aos atuais clientes (como estratégia de retenção de clientes). 

### 2.8. Parcerias Principais

&emsp; Este bloco descreve a rede de contatos da instituição, envolvendo fornecedores e parceiros que fazem o modelo de negócios funcionar. Aqui, pode-se classificar quatro tipos de parcerias diferentes [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas):

1. Alianças estratégicas entre não competidores;
2. Parcerias estratégicas entre competidores;
3. Empreendimentos em conjunto (Joint Ventures) para novos negócios;
4. Relação entre comprador e fornecedor (para garantir fornecimentos confiáveis).

&emsp; A estratégia de parcerias principais contempla, sobre regime de alianças estratégicas não competitivas, o apoio do Laboratório de Materiais para Produtos de Construção do IPT, para validação científica do sistema, além do fornecimento de *datasets* para treinamento do modelo preditivo e materiais de estudo científicos. Ainda, sobre regime do tópico 4 supracitado, considera-se parcerias com provedores de infraestrutura em nuvem, como AWS e Google Cloud.

### 2.9. Estrutura de Custos

&emsp; Neste bloco, incluem-se todos os custos envolvidos no modelo de negócios, ou seja, custos que se relacionam com a geração e entrega de [valor ao cliente](#21-proposta-de-valor), manutenção de [Relacionamento com Clientes](#24-relacionamento-com-clientes) e geração de receitas [(Osterwalder & Pigneur, 2010)](#referências-bibliográficas).

&emsp;A estrutura de custos será composta pelos investimentos em produção e manutenção da plataforma de software, desenvolvimento e hospedagem em nuvem, salários da equipe técnica, aquisição e manutenção de equipamentos de captura de imagem, licenciamento de tecnologias (se necessário), custos operacionais logísticos e aportes contínuos em pesquisa e desenvolvimento para manter a competitividade tecnológica do produto fornecido.

## 3. Referências Bibliográficas

1. OSTERWALDER, A.; PIGNEUR, Y. Business Model Generation. Ed. Wiley John & Sons. New Jersey – Estados Unidos, 2010. Disponível em: https://vace.uky.edu/sites/vace/files/downloads/9_business_model_generation.pdf. Acesso em: 24 de abril de 2025.

2. Termo de Abertura do Projeto do Inteli, TAPI. Acesso em: 25 de abril de 2025.

3. VICELLI, B.; TOLFO, C. Um estudo sobre valor utilizando o Business Model Canvas. Bagé, Rio Grande do Sul - Brasil, 2016. Disponível em: https://www.revistaespacios.com/a17v38n03/a17v38n03p14.pdf. Acesso em: 24 de abril de 2025.