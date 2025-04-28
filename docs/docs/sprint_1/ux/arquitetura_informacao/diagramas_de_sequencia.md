---
title: Diagramas de Sequência
sidebar_position: 3
---

&nbsp;&nbsp;&nbsp;&nbsp;Os diagramas de sequência, conforme descrito no artigo “Sequence Diagrams – Unified Modeling Language (UML)” [[1]](#referências), são ferramentas para modelar a interação entre objetos em um sistema ao longo do tempo. Desse modo, eles demonstram como mensagens ou eventos são trocados entre diferentes partes do sistema, proporcionando uma visualização clara e detalhada dos fluxos de execução. Assim, seu uso permite demonstrar como será a interação do usuário com as informações disponíveis na solução apresentada.

&nbsp;&nbsp;&nbsp;&nbsp;Embora sejam duas personas com focos distintos dentro do sistema, a dependência de seus fluxos justifica a representação em um único diagrama de sequência, conforme ilustrado na Figura 1. Assim, o diagrama unificado para as duas personas permite a  visualização das etapas individuais e também enfatiza a interconexão necessária para que o sistema atinja seu propósito de fornecer informações completas.  

<div align="center">
<sub>Figura 1 - Diagrama de Sequência</sub>

![Diagrama de Hierarquia das Páginas](</img/diagrama_sequencia.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Conforme pode ser visto na Figura 1, a interação se inicia com ambos os usuários, Maria e Rafael, realizando o processo de autenticação no sistema e, uma vez autenticados, seus fluxos se divergem, refletindo suas diferentes responsabilidades. Maria, inicia com a criação de um novo projeto, cujas informações também é responsável por preencher. Seu fluxo termina com a exportação do Relatório Final, que é o resultado do processamento das informações inseridas e analisadas no sistema.

&nbsp;&nbsp;&nbsp;&nbsp;Por outro lado, o fluxo de Rafael é focado na importação de imagens: ele utiliza a funcionalidade de Captura de Imagens por meio do drone e também realiza o Upload de Imagens de outras fontes. Assim, após o envio dessas imagens, o sistema realiza a análise delas, retornando as classificações das fissuras presentes nelas.

&nbsp;&nbsp;&nbsp;&nbsp;Assim, para que esses fluxos aconteçam da forma adequada, os elementos necessários no diagrama são:

- **Login**: Tela de autenticação do sistema, cuja responsabilidade é validar credenciais (e-mail e senha) e redirecionar os usuários para a tela com os projetos.
- **Projetos**: Tela de listagem e cadastro de projetos, cujo objetivo é exibir projetos existentes e permitir a criação de novos.
- **Projeto**: Tela de detalhes de um projeto específico, cuja função é armazenar informações como responsáveis, edifícios, localização e relatórios.
- **Imagem Drone**: Interface para visualização e captura de imagens em tempo real via drone.
- **Upload Imagem**: Tela para envio de imagens de outras fontes (ex.: câmeras de alta resolução).
- **Analisar Imagem**: tela que possibilita o acesso ao funcionamento do modelo de classificação das imagens, retornando os resultados.
- **Drone**: Equipamento físico para captura de imagens aéreas que transmite imagens em tempo real para a interface Imagem Drone.

&nbsp;&nbsp;&nbsp;&nbsp;Nesse cenário, considerando as personas e os participantes descritos acima, as etapas presentes no fluxo são:

   - **1.1**: Maria insere e-mail e senha na tela de Login.
   - **1.2**: O sistema valida as credenciais e a redireciona para a tela Projetos.
   - **1.3**: Maria cadastra um novo projeto na tela Projetos.
   - **1.4**: O sistema confirma a criação e exibe o projeto na lista.
   - **1.5**: Maria acessa a tela Projeto para adicionar detalhes (responsáveis, edifícios, localização).
   - **1.6**: Rafael insere suas credenciais na tela Login.
   - **1.7**: O sistema autentica e o redireciona para a tela Imagem Drone.
   - **1.8**: Rafael controla o Drone para posicioná-lo onde há fissuras para capturar.
   - **1.9**: O drone envia as imagens para a tela Imagem Drone.
   - **1.10**: Rafael visualiza as imagens em tempo real.
   - **1.11**: Rafael realiza a captura de uma imagem específica de uma fissura ao apertar um botão da tela Imagem Drone.
   - **1.12**: Rafael envia imagens de outras câmeras na tela Upload Imagem.
   - **1.13**: O sistema confirma o carregamento bem-sucedido.
   - **1.14**: O modelo classifica as fissuras e retorna os resultados a Rafael na tela Analisar Imagem.
   - **1.15**: A tela Projeto consolida os dados (incluindo classificações) e gera um relatório para Maria exportar.

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, à vista do apresentado, o diagrama de sequência detalha a interação de cada persona com o sistema, evidenciando tanto seus fluxos individuais quanto a interdependência de suas ações para a geração do relatório final. Dessa forma, essa representação visual facilita a compreensão do fluxo de informações dentro da aplicação e valida a cobertura das necessidades dos usuários identificados. Logo, as próximas etapas de desenvolvimento, como a criação dos wireframes, detalharão ainda mais a interface e a experiência do usuário, a fim de melhor expor como as informações serão apresentadas aos usuários.

## Referências 

[1] GEEKSFORGEEKS. Unified Modeling Language (UML) | Sequence Diagrams. Disponível em: https://www.geeksforgeeks.org/unified-modeling-language-uml-sequence-diagrams.

‌