# Metadesign

## Introdução

&emsp; O metadesign consiste em refletir e documentar os caminhos estratégicos, conceituais e técnicos utilizados no processo de desenvolvimento de uma solução. Nesta seção, são apresentados os principais elementos que fundamentaram o projeto, trazendo fatores de mercado, escolha de tecnologias e preocupações socioculturais e ambientais.[1]

&emsp; A documentação do metadesign busca registrar o que foi feito e justificar como e por que cada decisão foi tomada, promovendo uma visão crítica e ampla de design.


## Fatores marcadológicos

&emsp; O projeto foi desenvolvido com foco em atender uam demanda na espeção de prédios e fachadas, trazendo uma solução eficiente e de baixo custo, principalmente em instituições públicas e centros de pesquisa, como o IPT. O mercado da construção civil e manutenção predial busca soluções que otimizem o tempo, reduzam os riscos de inspeção manual e  aumentem a precisão técnica. 

&emsp; Além disso, o uso de drones e aprendizado de máquina é uma tendência crescente no setor, e o sistema proposto se posiciona como uma ferramenta escalável e adaptável a diversos  tipos de edificações e ambientes industriais, com potencial de expansão para o setor privado e prefeituras. 


## Sistema produto/design

&emsp; A solução desenvolvida foi pensada como um sistema integrado, composto por:

* Captura de imagens via drone ou upload de imagens pelo usuário
* Análise automatizada pelo modelo preditivo para detecção de rachaduras
* Interface web para visualização, aprovação e exportação de relatórios e outras informações do projeto
* Armazenamento e gestão de imagens em nuvem via Supabase

&emsp; O design foi pensado levando em consideração usabilidade, modularidade e clareza visual, garantindo que os usuários (engenheiros civis, técnicos de laboratório) possam interagir facilmente com os dados, realizar análises rápidas e tomar decisões com base nas informações apresentadas

## Sustentabilidade ambiental

&emsp; A proposta considera aspectos de sustentabilidade tanto no desenvolvimento quanto na implantação da solução:

* redução de deslocamentos físicos e inpeções presenciais, diminuindo a emissão de carbono relacionada ao transporte
* Uso de infraestrutura em nivem e ferramentas digitais que evitam desperdício de papel, impressão de relatórios e consumo de materiais descartáveis
* Uso de tecnologiais recarregáveis, com drones utilizados (como o DJI Inspire) que seguem padrões de recarregamento elétrico e podem ser reaproveitados para múltiplas aplicações

## Influências socioculturais

&emsp;   O projeto foi desenvolvido considerando o contexto institucional do IPT e as necessidades de técnicos e engenheiros que trabalham com segurança estrutural. A solução valoriza:

* A valorização da mão de obra técnica especializada, oferecendo uma ferramenta de apoio que potencializa a atuação humana, ao invés de substituí-la
* A proposta também se alinha com os objetivos de modernização da infraestrutura pública, contribuindo para políticas de prevenção e manutenção predial em instituições públicas


## Tipológico-formais e ergonômicos

&emsp; As interfaces de aplicação foram projetadas com foco em:

* Clareza visual, com uso de cores neutras e alto contraste para ambientes ensolarados ou com pouca iluminação (comum em inspeções de campo)
* Compatibilidade com dispositivos móveis, permitindo acesso rápido às informações durante a inspeção
* Layout responsivo e acessível, com botões grandes, navegação simplificada e feedback visual direto
* As funcionalidades seguem uma lógica sequencial (captura > análise > visualização > exportação), otimizando o fluxo de trabalho do usuário


## Tecnologia produtiva e materiais empregados

&emsp; A solução possui tecnologias modernas acessíveis:

* **Software:** Supabase, AWS, Figma, React/tailwind(web)
* **Hardware:** Drones com câmera de alta resolução (DJI), servidores cloud, computadores para desenvolvimento
* **Materiais:** Não há fabricação física direta no MVP


## Conclusão

&emsp; O metadesign do projeto mostra um processo de criação preocupado com a função prática, viabilidade mercadológica e responsabilidade sociocultural. As decisões tomadas ao longo do desenvolvimento foram dguiadas por critérios de usabilidade, sustentabilidade, integração tecnológica e alinhamento com as reais necessidads do parceiro (IPT).

&emsp; Ao estruturas o sistema de forma flexível e escalável, o projeto desenvolvido é uma solução com potencial para expansão para outras realidades similares. A documentação do metadesign contribui para o entendimento do projeto, que pode ser adaptado, aprimorado e reutilizado, reforçando o compromisso com o design consciente e evolutivo.


## Referências 

[1] WIKIPEDIA. Metadesign. Disponível em: https://en.wikipedia.org/wiki/Metadesign. Acesso em: 24 jun. 2025.