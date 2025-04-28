---
title: Arquitetura da Informação
sidebar_position: 1
---

&nbsp;&nbsp;&nbsp;&nbsp;Conforme detalhado na seção de Introdução, com a elaboração do projeto almeja-se realizar a coleta de imagens e processá-las de modo a retornar ao usuário algumas informações, como as imagens obtidas e o resultado da classificação. Assim, faz-se necessário estruturar e organizar a forma como essas informações serão apresentadas, a fim de garantir clareza e usabilidade para os usuários.

&nbsp;&nbsp;&nbsp;&nbsp;Desse modo, segundo o vídeo "O que é Arquitetura de Informação?" do canal UXNOW [1], a Arquitetura da Informação visa mapear como a informação ficará organizada dentro de uma aplicação. Ela representa como uma persona irá consumir os conteúdos dentro de um contexto específico de uso do sistema. Logo, esse mapeamento é essencial para garantir que os usuários consigam navegar e entender facilmente os dados apresentados.

&nbsp;&nbsp;&nbsp;&nbsp;No caso deste projeto, foram identificadas duas personas principais: Rafael, o técnico de laboratório que busca X informações, e a Maria, Engenheira Civil que busca X informações.

&nbsp;&nbsp;&nbsp;&nbsp;Dessarte, a disposição das informações deve considerar a jornada do usuário desde a coleta das imagens pelo robô até a visualização dos resultados processados. Dessa forma, na Figura 1 abaixo, é possível visualizar a hierarquia das páginas do sistema, a qual visa aprensertar todas as etapas dessa jornada.

<div align="center">
<sub>Figura 1 - Diagrama de Hierarquia das Páginas</sub>

![Diagrama de Hierarquia das Páginas](</img/diagrama_pags.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Logo, conforme pode ser observado na Figura 1, as seguintes telas estarão presentes no projeto: Login, Projetos (Visualização de todos os projetos cadastrados), Projeto (visualização de um projeto específico), Upload de Imagem (onde pode-se carregar imagens para análise dentro de um projeto), Imagens Drone (onde é possível ver as imagens do drone em tempo real e capturar a imagem), Analisar Imagem (Onde estão disponíveis as previsões do modelo para cada imagem), Relatório (conjunto dos dados de um projeto para serem exportados).

&nbsp;&nbsp;&nbsp;&nbsp;Ainda, duas formas de mapear essa arquitetura da informação é com a construção de diagramas de sequência e por meio de Wireframes, os quais estão disponíveis nas outras duas seções da pasta Arquitetura da Informação.
