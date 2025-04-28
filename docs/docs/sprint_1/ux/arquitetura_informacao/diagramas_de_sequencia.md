---
title: Diagramas de Sequência
sidebar_position: 2
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

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, à vista do apresentado, o diagrama de sequência detalha a interação de cada persona com o sistema, evidenciando tanto seus fluxos individuais quanto a interdependência de suas ações para a geração do relatório final. Dessa forma, essa representação visual facilita a compreensão do fluxo de informações dentro da aplicação e valida a cobertura das necessidades dos usuários identificados. Logo, as próximas etapas de desenvolvimento, como a criação dos wireframes, detalharão ainda mais a interface e a experiência do usuário, a fim de melhor expor como as informações serão apresentadas aos usuários.

## Referências 

[1] GEEKSFORGEEKS. Unified Modeling Language (UML) | Sequence Diagrams. Disponível em: https://www.geeksforgeeks.org/unified-modeling-language-uml-sequence-diagrams.

‌