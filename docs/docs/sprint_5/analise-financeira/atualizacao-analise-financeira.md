# Atualização da Análise Financeira


&emsp; Após revisão detalhada da análise financeira apresentada anteriormente, foi identificado um erro no cálculo dos impostos, que foram aplicados antes da inclusão da margem de lucros, o que resultaria em um valor final do projeto incorreto, prejudicando o equilíbrio financeiro do projeto. 

&emsp; Para garantir a sustentabilidade da proposta, atualizamos a documentação considerando que os impostos incidem também sobre o valor do lucro. Assim, o novo cáĺculo aplica corretamente a carga tributária sobre o valor total, garantindo que a margem de lucro seja de fato recebida, após o pagamento de todos os tributos.

&emsp; A seguir está apresentado os valores corrigidos


# Atualização da Análise Financeira- Solução final

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



**Custo total com pessoal : R$ 152.500** 




## Ferramentas e softwares 

&emsp; Para o desenvolvimento da solução, foi necessário o uso de diversas ferramentas e softwares que auxiliaram na organização, no design, no desenvolvimento e na colaboração da equipe. A seguir, detalhamos os principais:



Ferramenta/Software | Finalidade | Plano Utilizado | Custo por Pessoa | Nº de Pessoas | Duração | Custo Total
| - | - | - | - | - | - | -
Supabase | Backend e armazenamento | Free Tier | R$ 0,00 | 6 | 10 semanas | R$ 0,00
AWS (EC2 + S3) | Deploy da aplicação e imagens | Plano sob demanda (estimado) | - | - | 10 semanas | R$ 900,00
Figma | Design de interfaces | Pro | R$ 75,00/mês | 1 | 2,5 meses | R$ 187,50
GitHub | Controle de versão | Pro | R$ 24,00/mês | 6 | 2,5 meses | R$ 360,00
Trello | Organização de tarefas | Gratuito | R$ 0,00 | 6 | 10 semanas | R$ 0,00
VS Code | Editor de código | Gratuito | R$ 0,00 | 6 | 10 semanas | R$ 0,00
Slack | Comunicação da equipe | Gratuito | R$ 0,00 | 6 | 10 semanas | R$ 0,00

Considerando R$75,00 por mês para o desginer, único que necessitará o plano plano Pro do Figma.

**Total estimado com ferramentas e softwares: R$ 1.447,50**


## Equipamentos

&emsp; Embora os testes sejam realizados em ambiente virtual, é necessário um mínimo de hardware para o time técnico poder desenvolver e testar este projeto:

Item	|Descrição|	Valor (R$)
|-|-|-|
Notebooks para desenvolvimento |6 unidades	|35.000
Dispositivo Android e iOS	|Testes da aplicação mobile	|6.000
Roteador de campo 4G | Conexão em áreas externas | 2.500

**Total com equipamentos: R$ 43.500**



## Custos operacionais 

&emsp; Estimamos os custos indiretos com  energia, internet, alimentação, comunicação, suporte jurídico e contábil foram estimados como 25% do total dos custos fixos (pessoal + equipamentos + ferramentas/softwares):

**Custo base = R$ 152.500 + 1.447,50 + 43.500 = R$ 197.447,50**  

**Custo operacional = R$ 197.447,50 × 0,25 = R$ 49.361,88**



## Manutenção 

&emsp; Após a entrega da solução, será necessário garantir a estabilidade e o suporte ao sistema durante 12 meses.


Item | Descrição | Valor (R$)
| - | - | -
Servidores e banco de dados | AWS (t2.medium + S3) + Supabase extra | 18.000
Suporte técnico (meio período) | Dev + DevOps | 60.000
Atualizações, melhorias e testes | Correções e atualizações| 15.000
Licenças anuais | Figma Pro + GitHub Pro (1 ano) | 1.200


**Total com manutenção anual: R$ 94.200**


## Lucros e impostos

&emsp;   Para garantir uma margem de lucro líquido de 20% mesmo após o pagamento de impostos, utilizamos a fórmula reversa de precificação, considerando uma carga tributária de 17% (regime de Lucro Presumido, CNAE 6201-5/00).

**Valor final = (Custo total) / (1 - lucro - imposto)**

**Valor final = R$ 341.009,38 / (1 - 0,20 - 0,17) = R$ 541.284,72**


# Conclusão 

&emsp; A solução proposta representa um investimento de aproximadamente R$ 541 mil e engloba o desenvolvimento completo da aplicação (backend, frontend web/mobile e IA), um suporte técnico e manutenção anual e uma margem de lucro sustentável para a continuidade do projeto.  Este valor garante a entrega de uma solução funcional e adaptada à realidade do IPT, com potencial para expansão futura e reaproveitamento em outras campos que enfrentem desafios semelhantes. 