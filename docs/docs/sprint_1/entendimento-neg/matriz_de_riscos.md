---
title: Matriz de Riscos
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Matriz de riscos e oportunidades
&emsp;No decorrer do desenvolvimento de projetos, é natural que quaisquer empresas e grupos tenham que lidar com determinados riscos e perigos que possam ser obstáculos para seus objetivos finais. O que diferencia essas empresas e grupos é justamente a maneira como elas conseguem identificar quais podem ser esses riscos e como podem lidar com eles (Sartori, 2023)[[1]](https://qualyteam.com/pb/blog/matriz-de-risco-pgr). Assim, surge uma ferramenta de gestão simples e amplamente utilizada: a Matriz de Riscos, uma ferramenta que auxilia um time a avaliar o nível de risco levando em consideração a probabilidade do risco acontecer e o impacto caso aconteça. Além da Matriz de Riscos, existe também a matriz de oportunidades, que funciona de maneira semelhante, porém, ao invés de avaliar riscos e ameaças ao projeto, avalia pontos fortes que devem ser aproveitados bem como as melhores estratégias para gerar ganhos a partir disso.

&emsp;O presente projeto trata do desenvolvimento de um algoritmo para processamento digital de imagens (PDI) na identificação e classificação de fissuras em revestimentos de argamassa em fachadas de edifícios. É um projeto que utilizará uma gama de tecnologias que serão compostas não só de software, mas também de hardware. De acordo com o TAPI (Termo de Abertura de Projeto do Inteli), o principal objetivo deste projeto é sanar a necessidade de uma solução para auxiliar na detecção automática e monitoramento de fissuras em edificações, principalmente em revestimentos de argamassa, que frequentemente causam risco à integridade estrutural das construções e aumentam os custos com manutenção corretiva. Levando tudo isso em consideração, fica claro que existirão diversos riscos (mas também oportunidades) que podem ser explorados no decorrer do projeto. O quadro (Figura 1) abaixo lista tais riscos/oportunidades.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Matriz de Riscos e Oportunidades</strong></p>
  <img 
    src={useBaseUrl('/img/matriz_de_riscos.png')} 
    alt="Matriz de Riscos e Oportunidades" 
    title="Matriz de Riscos e Oportunidades" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

## Potenciais riscos

&emsp;Para os riscos mencionados, temos:
1. Baixa precisão do algoritmo de classificação e identificação das fissuras, algo muito factível de se ocorrer, uma vez que o dataset de imagens de fissuras fornecidos pelo parceiro de projeto pode não ser suficiente para treinar modelos de maneira eficiente. Dessa forma, é possível mitigar este risco através do uso de datasets públicos a fim de treinar os modelos com um volume maior de dados.
2. Qualidade inadequada das imagens capturadas pela câmera do drone utilizado, que captura imagens a uma resolução de 5MP. Tal resolução pode não ser suficiente para identificar algumas rachaduras. Assim, uma maneira de contornar este risco pode ser a correção e melhoramento das imagens através de software.
3. Resistência de usuários à adoção do sistema, que atualmente estão acostumados a pilotar o drone, tirar as fotos e as interpretar de maneira manual. Por conta disso, será necessário ter uma documentação completa, intuitiva e que possa convencer esses usuários de que a solução melhora o trabalho deles.
4. Dificuldades de escalabilidade pós-implantação, uma vez que teremos modelos preditivos consideravelmente grandes rodando em tempo real na nuvem, algo que pode rapidamente ficar pesado e custoso se feito em larga escala. Por conta disso, será necessário realizar, por exemplo, parcerias com empresas que oferecem serviços de Cloud a fim de reduzir custos de hospedagem de servidores. 

## Potenciais oportunidades 

&emsp;Já para as oportunidades, possuímos:
1. Parceria estratégica com fabricante de drones, uma vez que os drones utilizados representam, possivelmente, o maior custo do projeto. Assim, estabelecer uma parceria com empresas como a DJI (fabricante de drones) a fim de abaixar os preços em remessas maiores de pedidos pode diminuir consideravelmente o custo de uma implementação em larga escala do projeto.
2. Expansão para análise preditiva de fissuras é algo que é mencionado no TAPI (Termo de Abertura de Projeto Inteli) e que pode ser feito sem um esforço tão significativo através da implementação de modelos de Machine Learning que poderão, por exemplo, inferir como ocorrerá a evolução de uma fissura e qual será o melhor momento para realizar a manutenção.
3. Integração com drones autônomos para inspeções automatizadas, algo que pode agilizar o processo de encontrar fissuras e diminuir erros humanos e pode ser implementado através de rotas automatizadas e algoritmos de mapeamento e movimentação e ser vendido como um “módulo extra” para o produto. 

## Conclusão

&emsp;Por fim, fica claro que, ao olhar para uma matriz de riscos e oportunidades, empresas e equipes podem desenvolver estratégias mais eficazes para gerenciar seus riscos e aproveitar oportunidades de negócio (Nuzzi, 2021)[[2]](https://medium.com/@jonesroberto/matriz-de-risco-e-oportunidades-89a3d5f708ad). No caso do projeto atual, por exemplo, é possível identificar riscos como a falta de dados para treinamento de modelos de classificação e também como resolver este problema. Além disso, é possível também identificar oportunidades interessantes, como a possibilidade de fazer um modelo preditivo de evolução de fissuras, algo que agrega grande valor ao projeto. No geral, essa matriz é uma ferramenta que traz um entendimento mais aprofundado sobre diversos pontos de atenção que podem ser explorados no projeto. 

## Bibliografia

[[1]](https://qualyteam.com/pb/blog/matriz-de-risco-pgr) SARTORI, A. Matriz de risco: saiba como aproveitar essa ferramenta no PGR. Disponível em: https://qualyteam.com/pb/blog/matriz-de-risco-pgr. Acesso em: 23 abr. 2025.

[[2]](https://medium.com/@jonesroberto/matriz-de-risco-e-oportunidades-89a3d5f708ad) NUZZI, J. R. Matriz de Risco e Oportunidades - Jones Roberto Nuzzi - Medium. Disponível em: https://medium.com/@jonesroberto/matriz-de-risco-e-oportunidades-89a3d5f708ad. Acesso em: 23 abr. 2025.

