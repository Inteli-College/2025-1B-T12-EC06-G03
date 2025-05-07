---
title: User Stories
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# User Stories

&emsp; As user stories são descrições breves e objetivas que demonstram uma necessidade de um usuário em relação ao sistema, formuladas do ponto de vista das personas desenvolvidas. Elas ajudam no processo de desenvolvimento do sistema para melhor entender quem é o usuário, as necessidades dele e o que ele busca alcançar com a solução [(AGILE ALIANCE, s.d.)](https://www.agilealliance.org/glossary/user-stories/).

&emsp; Além disso, por serem centradas no usuário, as user stories orientam o desenvolvimento de funcionalidades que realmente atendam aos problemas e objetivos dos usuários finais, promovendo uma construção mais assertiva do produto final. 

&emsp; Neste projeto, foram desenvolvidas user stories com base nas duas personas definidas: Rafael Lima (Técnico de laboratório) e Maria Silva (Engenheira Civil). Cada user story está relacionada a uma funcionalidade do sistema, mostrando melhor como a nossa solução atende diretamente as necessidades identificadas nas personas. 

## User Story Rafael Lima

&emsp; As user stories desenvolvidas para Rafael são mais direcionadas a coleta de dados em campo por meio da pilotagem do drone e da captura de imagens das fissuras. Assim, suas user stories focam na interação direta com os equipamentos de coleta, além da utilização das funcionalidades do sistema voltadas para upload e organização dos registros, construindo uma base confiável de dados que será utilizada para análises futuras.

### User Story 1
|Item|Detalhamento|
|-|-|
|US01|Como técnico de laboratório, quero que a plataforma forneça uma interface simples e intuitiva para a visualização em tempo real e captura de imagens através da câmera do drone|
|CR01|Rafael deve conseguir acionar a câmera do drone e tirar fotos das fissuras quando necessário|
|CR02|A interface deve ser simples, com feedback visual em tempo real|
|Feature|Controle e captura de imagens|

### User Story 2
|Item|Detalhamento|
|-|-|
|US02|Como técnico de laboratório, quero que as imagens capturadas pelo drone sejam automaticamente organizadas e enviadas para a plataforma, para evitar erros com a gestão manual de dados|
|CR01|O sistema deve realizar o upload das imagens automaticamente|
|CR02|As imagens devem ser armazenadas de forma organizada e facilmente acessíveis na plataforma para análise posterior|
|Feature|Organização e upload das imagens|

## User Story Maria Silva

&emsp; As user stories de Maria refletem mais a sua atuação na etapa de pós-processamento das imagens, com foco na interpretação, classificação e priorização das fissuras. Seu trabalho depende diretamente da qualidade dos registros feitos em campo e necessita uma interface simples e eficiente para visualizar as imagens, analisar a gravidade das fissuras e gerar relatórios automatizados para a tomada de decisões embasadas. Assim, suas user stories destacam o uso das ferramentas de análise, dashboards e automação de relatórios, fundamentais para garantir a eficiência das ações de manutenção preventiva. 

### User Story 1
|Item|Detalhamento|
|-|-|
|US01|Como engenheira civil, quero acessar e visualizar as imagens das fachadas, com destaque para fissuras classificadas automaticamente|
|CR01|As imagens devem ser exibidas com marcações visuais nas áreas onde foram capturadas as fissuras|
|CR02|O sistema deve permitir a visualização clara das imagens, com possibilidade de ampliar para uma análise mais detalhada|
|Feature|Visualização das imagens das fissuras|

### User Story 2
|Item|Detalhamento|
|-|-|
|US02|Como engenheira civil, desejo que as fissuras capturadas sejam classificadas automaticamente para que eu possa tomar decisões de manutenção com base em dados|
|CR01|O sistema deve classificar automaticamente as fissuras com base em parâmetros técnicos pré-definidos|
|CR02|A classificação automática deve poder ser alterada pelo engenheiro, mantendo logs das alterações|
|Feature|Classificação automática das fissuras e logs de edição de classificação

## Conclusão 
&emsp; As User Stories desenvolvidas para as personas Rafael Lima e Maria Silva oferecem um entendimento claro das necessidades e objetivos específicos de cada usuário no contexto do sistema de monitoramento de fissuras em prédios, auxiliando no desenvolvimento de funcionalidades que atendem à expectativa do cliente final.

## Bibliografia 

1. AGILE ALLIANCE. User Stories. Disponível em: https://www.agilealliance.org/glossary/user-stories/. Acesso em: 28 abr. 2025.