---
title: Arquitetura Atualizada
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

&nbsp;&nbsp;&nbsp;&nbsp;Conforme documentado em [Controle do Drone com ESP32](/sprint_2/controle_drone/ESP32), a utilização do microcontrolador ESP32 não se mostrou a melhor solução para o projeto proposto. Dessa forma, optou-se pela adoção do microcomputador Raspberry Pi para resolver a questão da conectividade, cuja implementação será discutida na próxima seção desta documentação.

&nbsp;&nbsp;&nbsp;&nbsp;Nesse sentido, a arquitetura proposta foi atualizada para refletir a nova solução, conforme as conclusões obtidas. Logo, a Figura 1 abaixo apresenta o diagrama com as alterações realizadas.

<div align="center">
<sub>Figura 1 - Proposta de Arquitetura</sub>

![Tela de Projetos](</img/arquitetura_sprint2.png>)
<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Dessarte, o microcontrolador foi substituído pelo microcomputador na Arquitetura da Figura 1; e a comunicação entre o microcomputador e o drone será feita via wi-fi, enquanto a comunicação com o tablet será via internet com USB. Assim, à vista do apresentado, a arquitetura reflete a solução proposta para resolver a questão da conectividade dado que a solução pensada anteriormente foi considerada adequada.
