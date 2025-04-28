---
title: Diagramas de Relação
sidebar_position: 2
---

&nbsp;&nbsp;&nbsp;&nbsp;Segundo o artigo "The UML 2 class diagram" [[1]](#referências), um diagrama de relações UML (Unified Modeling Language) é representação visual do relacionamento de elementos entre si. Esses relacionamentos podem incluir herança, agregação, associação, dependência, entre outros. Assim, o diagrama de relações UML é uma ferramenta que permite o entendimento do design, da arquitetura e da implementação da proposta de um sistema. 

&nbsp;&nbsp;&nbsp;&nbsp;Nesse sentido, a Figura 1 mostra o diagrama de relação para as telas disponíveis na aplicação. Dessa forma, esse diagrama demonstra como seria o fluxo do usuário entre as telas e quais informações seriam mostradas para ele em cada uma dessas telas.

<div align="center">
<sub>Figura 1 - Diagrama de Relação</sub>

![Diagrama de Relação](</img/diagrama_relacao.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Conforme pode ser visto na Figura 1, o ponto de entrada do sistema é o Login, responsável pela autenticação dos usuários. Após a autenticação, o usuário é redirecionado para a seção de Projetos, a qual é organizada em "Projetos recentes" e "Todos os projetos". Além disso, cada Projeto é composto por informações como Empresa, Pessoas responsáveis, Localização e Edifícios. Ademais, a partir de um Projeto, é possível navegar para o Upload de Imagens, para a visualização da Imagem captuarda pelo Drone, para a verificação das classificações das imagens e para o Relatório.

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, o diagrama permite a visualização do fluxo de navegação entre as telas da aplicação, facilitando a compreensão da estrutura geral do sistema e das interações disponíveis para o usuário. Dado que mostra o caminho lógico entre as telas e destaca quais informações estarão disponíveis para o usuário.

## Referências

[1] IBM Developer. The UML 2 class diagram. Disponível em: https://developer.ibm.com/articles/the-class-diagram/#the-class-diagram-in-particular2.

‌