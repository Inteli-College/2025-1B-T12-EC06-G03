---
title: Requisitos Não Funcionais Atualizados
sidebar_position: 2
---

&nbsp;&nbsp;&nbsp;&nbsp;Assim como os Requisitos Funcionais se adaptam às novas informações descobertas ao longo do desenvolvimento de um projeto, os Requisitos Não Funcionais (RNFs) também precisam ser constantemente atualizados. Essa atualização busca garantir que o sistema continue a atender às expectativas de desempenho, usabilidade, segurança e outras qualidades essenciais, refletindo a evolução do projeto e as necessidades do usuário.

<div align="center">
<sup>Quadro 1 - Requisitos Não Funcionais</sup>

| ID   | Nome do RNF                                               | Métrica Principal                                          | Método de Verificação                                           |
| :--- | :-------------------------------------------------------- | :--------------------------------------------------------- | :-------------------------------------------------------------- |
| RNF1 | Qualidade da Transmissão de Vídeo                         | FPS ≥ 10; Latência ≤ 500 ms                                | Medição com OpenCV, simulação de variações de rede              |
| RNF2 | Acurácia do Modelo de Classificação                       | Acurácia ≥ 70%; variação ≤ 5%                              | Validação cruzada com subdivisões de imagens                    |
| RNF3 | Precisão na Detecção de Fissuras                          | Acurácia ≥ 90%; FP < 5%; FN < 7%                           | Avaliação em condições de iluminação variadas                   |
| RNF4 | Autonomia da Bateria do Drone                             | Voo contínuo ≥ 15 minutos                                  | Testes de voo até 20% de carga residual                         |
| RNF5 | Latência de Captura de Imagem                             | L95 ≤ 2s; Lmed ≤ 1,5s; Lmax ≤ 3s                           | Automação de capturas e análise de logs                         |
| RNF6 | Acurácia em variação de luminosidade                      | Acurácia ≥ 65% em todas as faixas; FN ≤ 10%                | Comparação das métricas entre diferentes níveis de luminosidade |
| RNF7 | Tempo de Processamento por Imagem                         | RT90 ≤ 10s; Média ≤ 8s; Máximo ≤ 12s                       | Teste de processamento de batches de imagens                    |
| RNF8 | Feedback Visual Ações Críticas do Usuário | Presença de sinais visuais em eventos críticos | Testes de usabilidade e validação com usuários                  |
| RNF9 | Armazenamento Seguro de Senhas                            | Hash com algoritmos como bcrypt, Argon2 ou PBKDF2          | Inspeção de código e auditoria de segurança                     |

<sup>Fonte: Material Produzido pelos autores. (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Como ilustrado no Quadro 1, a versão atualizada dos Requisitos Não Funcionais abrange pontos importantes para aprimorar a experiência do usuário e a robustez do sistema. O RNF8 foi incluído para aprimorar a usabilidade com feedback visual em momentos críticos, enquanto o RNF9 reforça a segurança do usuário através do uso de algoritmos de hash para o armazenamento de senhas. Manter os RNFs alinhados ao progresso do projeto é fundamental para a entrega de um sistema de alta qualidade e confiabilidade.
