---
title: Requisitos Funcionais Atualizados
sidebar_position: 1
---

&nbsp;&nbsp;&nbsp;&nbsp;Ao longo do desenvolvimento do projeto, novas informações são trazidas pela empresa parceira e novas descobertas são feitas pelo grupo. Nesse sentido, os requisitos funcionais de um projeto devem ser constantemente revisados e ajustados. Dessa forma, essa prática garante que o produto final esteja alinhado com a viabilidade técnica e, mais importante, que atenda às necessidades e desafios reais do parceiro.

&nbsp;&nbsp;&nbsp;&nbsp;Nesse cenário, mostrou-se a necessidade de atualizar os requisitos funcionais para refletir as capacidades atuais e as prioridades estabelecidas. Desse modo, com base nos dados disponíveis, por exemplo, determinou-se que a identificação da gravidade das fissuras e sua localização exata dentro da edificação não seriam viáveis. Dado que essas funcionalidades exigiriam a implementação de ferramentas adicionais e foram postergadas para uma fase futura, pois não se alinhavam com o escopo e as prioridades do projeto. Dessarte, o Quadro 1 detalha os requisitos funcionais revisados.

<div align="center">
<sup>Quadro 1 - Requisitos Funcionais</sup>

| RF#  | Descrição | Regra de negócio |
|------|-----------|------------------|
| RF01 | O sistema deve integrar-se com drones ou câmeras de alta resolução para receber imagens automaticamente. | A integração deve permitir o envio automatizado de imagens capturadas em campo, reduzindo a necessidade de intervenção manual e aumentando a agilidade do processo. |
| RF02 | O sistema deve detectar automaticamente fissuras nas imagens recebidas. | A detecção deve ocorrer com base em técnicas de visão computacional e IA treinadas para identificar padrões típicos de fissuras em revestimentos. |
| RF03 | O sistema deve gerar relatórios automáticos com as fissuras detectadas. | Cada relatório deve conter informações como data, localização da fissura, gravidade e recomendações de manutenção. |
| RF04 | O sistema deve manter um histórico de detecções para monitoramento da evolução das fissuras. | O histórico deve ser armazenado por edificação e permitir comparações temporais entre diferentes coletas. |
| RF05 | O sistema deve permitir a visualização gráfica das fissuras detectadas sobrepostas às imagens originais. | O usuário deve conseguir identificar a localização exata de cada fissura de forma intuitiva. |
| RF06 | O sistema deve possuir uma interface gráfica responsiva e acessível via desktop. | A interface deve ser intuitiva e acessível para engenheiros civis e técnicos de manutenção predial. |
| RF07 | O sistema deve possibilitar o cadastro de edificações e áreas monitoradas. | Cada edificação deve conter seus próprios registros e imagens, organizados por data e localização. |
| RF08 | O sistema deve armazenar as imagens e dados analisados em uma base de dados segura. | A base de dados deve permitir acesso rápido e garantir integridade e confidencialidade das informações. |
| RF09 | O sistema deve permitir autenticação de usuários com diferentes níveis de permissão. | Usuários administrativos terão acesso total, enquanto operadores terão acesso restrito às funcionalidades básicas. |
| RF10 | O sistema deve permitir a validação manual de fissuras detectadas automaticamente. | O usuário técnico poderá revisar, editar ou descartar fissuras identificadas pela IA, garantindo maior confiabilidade na análise. |
| RF11 | O sistema deve registrar logs de atividades dos usuários. | Todas as ações críticas, como detecção, edição, geração de relatórios ou exclusão de dados, devem ser registradas com data/hora e identificação do responsável. |

<sup>Fonte: Material Produzido pelos autores. (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Logo, as revisões realizadas, como a exclusão da identificação da gravidade e localização exata das fissuras, demonstram a importância de priorizar funcionalidades viáveis dentro do escopo e dos recursos disponíveis. Assim, o Quadro 1 reflete esses ajustes ao apresentar requisitos que atendem às expectativas iniciais, além de incorpor os aprendizados e feedbacks obtidos ao longo do processo.