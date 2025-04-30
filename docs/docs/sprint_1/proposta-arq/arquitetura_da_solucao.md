---
title: Arquitetura da Solução
sidebar_position: 1
---

import useBaseUrl from '@docusaurus/useBaseUrl';

### Definição

&emsp;Arquitetura de software é o desenho estruturado que define como os diferentes componentes de um sistema interagem, quais responsabilidades cada parte assume e de que forma os dados circulam entre elas. Ela descreve padrões de comunicação, tecnologias empregadas, princípios de organização de código e regras que orientam decisões futuras de desenvolvimento. Bass, Clements e Kazman a descrevem como “a estrutura ou estruturas do sistema, que compreende elementos de software, propriedades externamente visíveis desses elementos e os relacionamentos entre eles” [(BASS; CLEMENTS; KAZMAN, 2003)](#ref1). Esta visão enfatiza que a arquitetura vai além de simples diagramas, constituindo um bloco de decisões fundamentais sobre a organização do sistema. Pensar nesse conjunto de fatores desde o início permite alinhar objetivos de negócio, requisitos técnicos e experiência do usuário em uma mesma visão compartilhada pela equipe.

&emsp;Ao documentar camadas, interfaces, protocolos e políticas de segurança, a arquitetura fornece um mapa claro que facilita a colaboração entre desenvolvedores, reduz riscos de retrabalho e favorece a evolução contínua do produto. Shaw e Garlan, pioneiros no estudo formal da arquitetura de software, destacam que a arquitetura proporciona uma visão holística do sistema, facilitando a compreensão de seus componentes e interações [(SHAW; GARLAN, 1996)](#ref2). Esta perspectiva reforça o papel da arquitetura como um meio de comunicação entre todos os envolvidos no desenvolvimento do software.

&emsp;Além disso, uma boa arquitetura suporta escalabilidade, desempenho consistente e manutenção previsível, pois antecipa pontos de acoplamento e distribui responsabilidades de maneira coerente com os requisitos não funcionais. Deste modo, ela serve como fundação para o sistema crescer de forma saudável, acompanhando demandas de usuários e mudanças tecnológicas sem comprometer qualidade ou prazos.

## Visão Geral

&emsp;O sistema pretende oferecer ao IPT uma plataforma única na qual os pesquisadores possam cadastrar e gerenciar projetos de inspeção, enviar imagens previamente obtidas ou capturá-las remotamente durante o voo, processar e classificar fissuras conforme os requisitos de desempenho, gerar relatórios em PDF no modelo exigido pelo laboratório, exportar seus dados em planilha e manter um histórico completo de projetos, tudo em uma interface intuitiva acessível pelo navegador.

## Macrocomponentes

| Camada               | Tecnologia Principal         | Responsabilidades                                                                 | Integrações Necessárias                       |
|----------------------|------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------|
| front-end            | React + Vite (TypeScript)    | interface responsiva, painel de projetos, player de vídeo em tempo real, formulários de upload, botão de captura | WebSocket para streaming, RESTful API, autenticação JWT, backend |
| api / back-end       | Java 17 + Spring Boot        | regras de negócio, controle de acesso, orquestração do fluxo de processamento, geração de relatórios, trilha de auditoria | PostgreSQL, serviço de processamento, serviço drone |
| serviço de processamento | Java 17 + OpenCV             | pré-processamento, detecção e classificação de fissuras, exposição de API assíncrona, enfileiramento de jobs | back-end |
| serviço drone        | Java 17 + SDK DJI            | recepção de comandos de captura, confirmação de foto armazenada, publicação de streaming via WebRTC/MQTT | back-end |
| persistência         | PostgreSQL 15                | usuários, projetos, metadados, logs de auditoria                                   | —                                             |
| fila de mensagens | RabbitMQ                     | desacoplamento das requisições de processamento                                    | back-end ⇄ serviço de processamento           |

&emsp;Essa tabela resume os macrocomponentes da solução, indicando para cada camada a tecnologia escolhida, as principais responsabilidades operacionais e as integrações que garantem o trânsito seguro de dados entre os módulos. Ela evidencia a separação de preocupações: o front-end lida somente com a experiência do usuário, o back-end centraliza a lógica e regras de negócio, o serviço de processamento cuida da inteligência artificial (CV) e o serviço drone faz a ponte com o hardware, enquanto os componentes de infraestrutura sustentam persistência, escalabilidade e desempenho.

## Diagrama

&emsp;O diagrama abaixo ilustra uma visão simplificada da arquitetura física do sistema, destacando o fluxo de dados entre os principais componentes envolvidos na operação em campo e na infraestrutura digital.

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Diagrama da arquitetura do projeto Athena</strong></p>
  <img 
    src={useBaseUrl('/img/diagrama_arquitetura.png')}
    alt="Diagrama da Arquitetura" 
    title="Diagrama da Arquitetura" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaboração própria (2025)</p>
</div>

&emsp;A comunicação inicia com o colaborador, que utiliza um tablet para interagir com o sistema de forma prática e portátil. O tablet se conecta a um microcontrolador via USB ou BLE (Bluetooth Low Energy). Esse microcontrolador atua como ponte de comunicação entre o tablet e o drone, recebendo comandos e repassando instruções de controle via Wi-Fi, como por exemplo capturar imagens. Ao realizar a captura, o drone envia os dados brutos para o sistema. Esses dados podem ser posteriormente processados e sincronizados com o restante da infraestrutura digital via protocolo HTTP, conectando o tablet ao sistema web.

&emsp;No ambiente digital, os colaboradores também acessam a plataforma por meio de um front-end web que se comunica com o back-end por meio de requisições HTTP. O back-end é responsável por orquestrar as operações e regras de negócio, processar dados e interagir com o banco de dados relacional através de queries SQL. Banco esse, que garante a persistência das informações, o histórico de operações e a integridade dos projetos registrados na plataforma.

&emsp;Este modelo reflete a integração entre hardware e software, garantindo que os dados capturados em campo sejam corretamente processados, armazenados e visualizados pela equipe técnica. A separação entre os fluxos de coleta, processamento e apresentação assegura flexibilidade, segurança e desempenho, importantes para o uso em ambientes de pesquisa aplicada e inspeções técnicas.

### Fluxo Principal “capturar e classificar”

&emsp;Quando o pesquisador, a partir da interface react, pressiona o botão de captura, o front-end envia uma requisição ao back-end que, por sua vez, encaminha o comando ao serviço drone. O drone realiza a fotografia, devolve uma confirmação e transmite o arquivo bruto para o object storage (ou semelhante). Em seguida, o back-end publica o identificador dessa imagem na fila RabbitMQ; o serviço de processamento consome a mensagem, executa a cadeia de pré-processamento, detecção, classificação e envia para o back-end. Finalmente, o back-end atualiza o status do projeto e envia via websocket a nova imagem já marcada à plataforma web, permitindo que o pesquisador acompanhe o diagnóstico das fissuras no projeto.

### Relatórios e Exportação

&emsp;A geração de relatórios será realizada diretamente no back-end: para exportações em planilha, a aplicação utilizará a biblioteca Apache POI (ou semelhante) para compor arquivos XLSX com as colunas padronizadas pelo IPT; para documentos em PDF, será empregado o iText (ou semelhante), que aplicará o template institucional, incluindo logotipo, sumário e tabelas de fissuras. Todo arquivo gerado será referenciado na tabela de relatórios do banco, garantindo rastreabilidade e fácil recuperação pelos usuários.

## Conclusão

&emsp;A arquitetura proposta se mostra coerente com os objetivos científicos e operacionais do IPT, oferecendo uma plataforma integrada que abrange desde a obtenção remota de imagens por drone até a geração automatizada de relatórios técnicos.

&emsp;A adoção de React no front-end assegura uma experiência de usuário fluida e responsiva, enquanto o ecossistema Java — Spring Boot no back-end e OpenCV no serviço de processamento — propicia uniformidade tecnológica, desempenho consistente e facilidade de manutenção. A separação clara de responsabilidades entre as camadas, aliada a mecanismos de mensageria e persistência confiáveis, viabiliza escalabilidade horizontal, reduz acoplamentos indesejados e procura manter o tempo de processamento conforme os limites estabelecidos pelos requisitos não funcionais.

&emsp;Além de atender às demandas imediatas de detecção e classificação de fissuras, a solução pavimenta o caminho para futuras extensões, como novos algoritmos de visão computacional ou integrações com dispositivos adicionais, sem impacto disruptivo sobre o núcleo do sistema. Ao utilizar-se de princípios arquiteturais consolidados, tecnologias amplamente adotadas e boas práticas de engenharia, o projeto buscará, em sua arquitetura, uma base para evoluir em consonância com as necessidades do laboratório, preservando qualidade, rastreabilidade e eficiência ao longo de todo o ciclo de vida da aplicação.

### Referências Bibliográficas

1. <span id="ref1">BASS, L.; CLEMENTS, P.; KAZMAN, R.</span> *Software Architecture in Practice*. Boston: Addison Wesley, 2003.

2. <span id="ref2">SHAW, M.; GARLAN, D.</span> *Software architecture: perspectives on an emerging discipline*. Prentice-Hall, 1996.  

3. <span id="ref3">GONÇALVES, Marcelo.</span> *Arquitetura e engenharia de software*. Medium, 2021. Disponível em: [https://medium.com/@marcelomg21/arquitetura-e-engenharia-de-software-3390030f22d3](https://medium.com/@marcelomg21/arquitetura-e-engenharia-de-software-3390030f22d3). Acesso em: 29 abr. 2025.

4. Termo de Abertura do Projeto do Inteli, TAPI. Acesso em: 25 de abril de 2025.