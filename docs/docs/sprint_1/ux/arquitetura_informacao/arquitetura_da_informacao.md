---
title: Arquitetura da Informação
sidebar_position: 1

---

&nbsp;&nbsp;&nbsp;&nbsp;Conforme detalhado na seção de [Introdução](/introducao), com a elaboração do projeto almeja-se realizar a coleta de imagens e processá-las de modo a retornar ao usuário algumas informações, como as imagens obtidas e o resultado da classificação. Assim, faz-se necessário estruturar e organizar a forma como essas informações serão apresentadas, a fim de garantir clareza e usabilidade.

&nbsp;&nbsp;&nbsp;&nbsp;Desse modo, segundo o vídeo "O que é Arquitetura de Informação?" do canal UXNOW [(UXNOW, 2016)](#referências-bibliograficas), a Arquitetura da Informação visa mapear como a informação ficará organizada dentro de uma aplicação, ou seja, ela representa como uma persona irá consumir os conteúdos dentro de um contexto específico de uso do sistema. Logo, esse mapeamento é essencial para garantir que os usuários consigam navegar e entender facilmente os dados apresentados.

&nbsp;&nbsp;&nbsp;&nbsp;No contexto deste projeto, foram identificadas duas [Personas](/sprint_1/ux/personas) principais: Rafael, o técnico de laboratório, e a Maria, Engenheira Civil. O objetivo de Rafael com o uso da solução é inserir as imagens capturadas pelo drone para que elas sejam classificadas pelo sistema; enquanto Maria almeja cadastrar novos projetos e exportar os relatórios prontos.

&nbsp;&nbsp;&nbsp;&nbsp;Dessarte, a disposição das informações deve considerar as jornadas desses dois usuários do início ao fim. Para tal, primeiramente, apresenta-se na na Figura 1 abaixo a hierarquia das páginas do sistema, cujo objetivo é apresentar todas as etapas dessas jornadas. Além desse diagrama de hierarquia foram utilizadas outras três metologias para realizar o mapeamento da arquitetura da informação: os [Diagramas de Sequência](/sprint_1/ux/arquitetura_informacao/diagramas_de_sequencia) e os Diagramas de Relação.

<div align="center">
<sub>Figura 1 - Diagrama de Hierarquia das Páginas</sub>

![Diagrama de Hierarquia das Páginas](</img/diagrama_pags.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Logo, conforme ilustrado na Figura 1, o projeto contemplará as seguintes telas: Login, Projetos (visualização de todos os projetos cadastrados), Projeto (detalhamento de um projeto específico), Upload de Imagem (para envio de imagens a serem analisadas dentro de um projeto), Imagens Drone (exibição em tempo real das imagens capturadas pelo drone, com opção de captura), Analisar Imagem (apresentação das previsões do modelo para cada imagem) e Relatório (consolidação dos dados de um projeto para exportação). Destarte, infere-se que a construção do diagrama de hierarquia permite mostrar se todas as etapas da jornada do usuário serão abrangidas pelas informações disponíveis no sistema.

&nbsp;&nbsp;&nbsp;&nbsp;Outrossim, a Figura 2 mostra o diagrama de relação para as telas disponíveis na aplicação. Dessa forma, esse diagrama demonstra como seria o fluxo do usuário entre as telas e quais informações seriam mostradas para ele em cada uma dessas telas.

<div align="center">
<sub>Figura 2 - Diagrama de Relação</sub>

![Diagrama de Relação](</img/diagrama_relacao.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Assim como pode ser visto na Figura 2, o ponto de entrada do sistema é o Login, responsável pela autenticação dos usuários. Após a autenticação, o usuário é redirecionado para a seção de Projetos, a qual é organizada em "Projetos recentes" e "Todos os projetos". Além disso, cada Projeto é composto por informações como Empresa, Pessoas responsáveis, Localização e Edifícios. Ademais, a partir de um Projeto, é possível navegar para o Upload de Imagens, para a visualização da Imagem captuarda pelo Drone, para a verificação das classificações das imagens e para o Relatório.

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, o diagrama permite a visualização do fluxo de navegação entre as telas da aplicação, facilitando a compreensão da estrutura geral do sistema e das interações disponíveis para o usuário. Dado que mostra o caminho lógico entre as telas e destaca quais informações estarão disponíveis para o usuário.

## Referências Bibliográficas

1. UXNOW / DANIEL FURTADO. O que é Arquitetura de Informação? //UXNOW. Disponível em: https://www.youtube.com/watch?v=vmvSMYaV4oE