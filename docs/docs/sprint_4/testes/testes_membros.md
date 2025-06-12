---
title: Testes com os Desenvolvedores
sidebar_position: 5
---

# Documentação dos Testes Técnicos com os Desenvolvedores 

#### Documentação dos Testes Técnicos com os Desenvolvedores

**Planilha de Registro Utilizada:**
[📎 Clique aqui para acessar a planilha dos testes](https://docs.google.com/spreadsheets/d/108u0t_NM4EJ6eYzEGlCkQBQTEWgIy26T/edit?usp=sharing&ouid=103007246092712924175&rtpof=true&sd=true)

## Introdução

&emsp;Durante a Sprint 4, foi conduzida uma rodada completa de testes técnicos com os próprios desenvolvedores do projeto. Esses testes complementam os testes de usabilidade com usuários externos, focando em validar o cumprimento dos **Requisitos Funcionais (RF)** e **Requisitos Não Funcionais (RNF)** da solução, com especial atenção à estabilidade do sistema, desempenho dos modelos de IA, segurança dos dados e integridade das funcionalidades.

&emsp;Realizar esse tipo de validação interna com os membros do próprio grupo é uma etapa essencial no ciclo de desenvolvimento. Por conhecerem profundamente o código e os fluxos implementados, os desenvolvedores conseguem executar os testes com olhar técnico e crítico, identificando falhas que poderiam passar despercebidas por usuários finais ou leigos. Além disso, a participação dos próprios desenvolvedores nesse processo é estratégica. Embora cada membro tenha atuado em módulos específicos do projeto, os testes promovem uma visão sistêmica, permitindo que todos compreendam o fluxo completo — da coleta de imagens à geração final de relatórios. Logo, essa etapa favorece a documentação rigorosa dos critérios de aceite e orienta decisões técnicas nas próximas sprints.

## Metodologia

&emsp;Cada integrante do grupo ficou responsável por executar os testes definidos nos seguintes eixos:

| ID  | Descrição                                                                |
| --- | ------------------------------------------------------------------------ |
| T01 | Verificação da qualidade da transmissão de vídeo (FPS e latência)        |
| T02 | Acurácia do modelo de detecção de fissuras                               |
| T03 | Verificação da completude dos relatórios gerados                         |
| T04 | Armazenamento e navegação correta do histórico de detecções              |
| T05 | Precisão da sobreposição das fissuras detectadas                         |
| T06 | Segurança e integridade da base de dados (inclusive simulação de ataque) |
| T07 | Registro de logs das ações críticas                                      |

&emsp;Os testes foram realizados pelos sete membros do grupo: **Carolina, Caio, Cecília Galvão, Sophia, Heitor, Gabriel e Matheus Jorge**, cada um executando os testes com instâncias reais do sistema e cenários variados, registrando métricas, capturas e comentários técnicos para posterior análise.

## Resultados e Observações

&emsp;Os testes permitiram identificar os seguintes pontos:

### Funcionalidades bem validadas:

* **T02 – Acurácia do Modelo:** Em todos os testes realizados, a acurácia do modelo superou 70%, com casos acima de 90%. O comportamento foi consistente e a variação entre execuções foi pequena, reforçando a confiabilidade do classificador.
* **T04 – Histórico:** A navegação e o armazenamento das imagens processadas funcionaram corretamente para todos os testadores. Logs de alteração foram registrados de forma clara, e o sistema manteve os dados organizados mesmo após múltiplas interações.
* **T07 – Registro de Logs:** Todos os logs de ações críticas (geração de relatório, edição, validação manual etc.) foram registrados corretamente com ID, data e hora, permitindo rastreabilidade adequada.

