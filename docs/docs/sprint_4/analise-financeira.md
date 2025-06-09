---
title: Análise Financeira do Projeto
sidebar_position: 1
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Análise Financeira- Solução final
&emsp; A solução final tem como propósito realizar uma aplicação completa e viável para uso cotidiano pelo IPT. Dessa forma, é necessário levar em conta uma série de fatores que influenciam o custo de implementação e operação da solução. Assim, esta documentação apresenta uam estimativa financeira detalhada de todos os custos envolvidos no projeto. O drone e infraestrutura física do IPT não foram incluídos na análise por já fazerem parte do ambiente e do projeto.


## Profissionais envolvidos 


&emsp; Para entregar a solução em 10 semanas, estimamos uma equipe diversa, com profissionais trabalhando em tempo integral:


Cargo|	Quantidade	|Valor Mensal por Pessoa (R$)	 | Total para o período
|- |- |- |-
Engenheiro(a) de Visão Computacional	| 1| 	13.000	|32.500
Desenvolvedor(a) Backend	|1	|9.000	|22.500
Desenvolvedor(a) Frontend (Web e Mobile)	|1	|9.500|	23.750
UX/UI Designer	|1	|8.000	|20.000
DevOps / Infraestrutura|	1	|11.500	|28.750
Gerente de Produto (PO)|	1	|10.000	|25.000

* **Engenheiro:** Responsável por desenvolver os algoritmos de detecção de rachaduras nas imagens capturadas pelo drone. Atua na escolha de modelos, pré-processamento de imagens, treinamento e validação da IA. Também colabora na integração da IA ao backend do sistema.

* **Desenvolvedor backend:** Desenvolve e mantém a lógica do servidor, APIs e banco de dados da aplicação. É responsável por conectar o frontend ao modelo, gerenciar o armazenamento das imagens, garantir segurança e escalabilidade da infraestrutura de dados.

* **Desenvolvedor frontend:** Cria as interfaces visuais da aplicação, tanto para web quanto para mobile. Trabalha em conjunto com o designer para implementar as telas e garantir boa experiência de uso, responsividade e funcionalidade. Também conecta essas interfaces ao backend.

* **UX/UI designer:** Responsável por desenhar as interfaces (layout, botões, navegação, etc.) e garantir uma experiência intuitiva e acessível ao usuário. Realiza protótipos, testes de usabilidade e documenta os fluxos de uso para facilitar o trabalho dos desenvolvedores.

* **DevOps/Infraestrutura:**  Cuida do ambiente de desenvolvimento, testes e produção. Configura servidores, pipelines de deploy, monitora a performance do sistema e garante sua disponibilidade. Também é responsável por segurança, backups e automações técnicas.

* **Gerente de Produto:** Coordena o projeto como um todo, garantindo que os objetivos do cliente (IPT) sejam atendidos. Define prioridades, acompanha prazos, facilita a comunicação entre as áreas e assegura que a entrega esteja alinhada às necessidades e expectativas do parceiro.



**Custo total com pessoal : R$ 152.000** 




## Ferramentas e softwares 

&emsp; Para o desenvolvimento da solução, foi necessário o uso de diversas ferramentas e softwares que auxiliaram na organização, no design, no desenvolvimento e na colaboração entre os membros da equipe. Algumas dessas ferramentas são gratuitas, enquanto outras possuem planos pagos que foram considerados no cálculo dos custos. A seguir, detalhamos os principais softwares utilizados:


Ferramenta/Software|	Finalidade	|Plano Utilizado	|Custo por Pessoa	|Nº de Pessoas	|Duração (semanas)	|Custo Total
|-|-|-|-|-|-|-|
Supabase|	Backend e armazenamento de imagens|	Gratuito (Free Tier)|	R$0,00|	6	|10	|R$0,00
Figma Pro	|Design de interfaces	|Pago (Pro)	|R$75,00/mês|	6|	~2,5 meses 	|R$1.125,00
Trello	|Organização de tarefas|	Gratuito	|R$0,00|	6	|10|	R$0,00
GitHub	|Controle de versão	|Gratuito	|R$0,00|	6	|10|	R$0,00
VS Code	|Edição de código|	Gratuito|	R$0,00	|6	|10|	R$0,00
Slack|	Comunicação da equipe|	Gratuito	|R$0,00	|6	|10	| R$0,00

Considerando R$75,00 por mês para o desginer, único que necessitará o plano plano Pro do Figma.

**Total estimado com ferramentas e softwares: R$187,50**


## Equipamentos

&emsp; Embora os testes sejam realizados em ambiente virtual, é necessário um mínimo de hardware para o time técnico poder desenvolver e testar este projeto:

Item	|Descrição|	Valor (R$)
|-|-|-|
Estações de trabalho (notebooks)	|7 unidades	|35.000
Dispositivo Android e iOS	|Testes da aplicação mobile	|6.000
Periféricos e rede (switch, roteador)	|Infraestrutura básica	|2.000

**Total com equipamentos: R$ 43.000**



## Custos operacionais 

&emsp; Custos indiretos com  energia, internet, alimentação, comunicação, suporte jurídico e contábil foram estimados como 25% do total dos custos fixos (pessoal + equipamentos + ferramentas/softwares):

Custo base = 152.00 + 187,50 + 43.000 = 195.187,50

Custo operacional = R$ 195.187,50* 0,25 = R$ 48.795.87



## Impostos

&emsp; Considerando o regime tributário Lucro Presumido (CNAE 6201-5/00), a carga tributária efetiva é de aproximadamente 17% sobre o valor bruto da operação.

Custo com impostos = (Custo base + operacionais) + 17% 
= 202.507,03 + 17% ≈ R$ 236.933,19


## Manutenção 

&emsp; Após a entrega da solução, será necessário garantir a estabilidade e o suporte ao sistema durante 12 meses.


Item	|Descrição	|Valor (R$)
|-|-|-
Servidores e banco de dados	|AWS, Supabase, armazenamento	|18.000
Suporte técnico (meio período)|	Dev + DevOps alocados	|60.000
Atualizações, melhorias e testes|	Correções e evolução incremental|	15.000
Licenças anuais|	Renovação de ferramentas	|800


**Total com manutenção anual: R$ 93.800**


## Lucros
&emsp; Aplicando uma margem de 20% sobre o valor total (projeto + manutenção):

Custo total = 243.984,37 + 93.800 = R$ 337.784,37

Valor final com lucro = 337.784,37 + 20% = R$ 405.341,24


# Conclusão 

&emsp; A solução proposta representa um investimento de aproximadamente R$ 405 mil e engloba o desenvolvimento completo da aplicação (backend, frontend web/mobile e IA), um suporte técnico e manutenção anual e uma margem de lucro sustentável para a continuidade do projeto.  Este valor garante a entrega de uma solução robusta, funcional e adaptada à realidade do IPT, com potencial para expansão futura e reaproveitamento em outras campos que enfrentem desafios semenlhantes. 