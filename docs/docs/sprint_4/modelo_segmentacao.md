---
title: Primeira versão do modelo de Segmentação de Fissuras
sidebar_position: 3
---

# Introdução

&emsp;Havendo desenvolvido um modelo de segmentação na *Sprint 3*, na *Sprint 4* o grupo Athena decidiu trabalhar na melhoria do modelo já desenvolvido, além de testar uma nova abordagem de fluxo de segmentação, como explicado neste documento.

# Análise do modelo já desenvolvido

&emsp;O modelo de segmentação desenvolvido na *Sprint 3*, apesar de apresentar métricas de classificação satisfatórias, ainda pecava na segmentação das fissuras identificadas na imagem.

&emsp; O atual fluxo de treinamento do modelo de segmentação segue os passos:

1. Input das imagens de treinamento;
2. Geração de máscaras (imagens binarizadas que destacam as fissuras);
3. Geração de labels (coordenadas dos polígonos que definem ao modelo as estruturas de cada tipo de fissura);
4. Treinamento do modelo **YOLOv8-seg** (através das labels).

&emsp; Foram estudadas diversas maneiras alternativas de endereçar a baixa fidelidade quanto à segmentação das fissuras nas imagens e, em conclusão das análises, identificou-se que o gargalo de treinamento do modelo **YOLOv8-seg** era o *script* de geração das máscaras para o processo de *labelling* das imagens de treinamento.

&emsp; Portanto, trabalhamos na melhoria do *script* de geração de máscaras, como explorado abaixo:

## Script de geração de máscaras

&emsp; gabs explica aqui!!!

# Novo fluxo de segmentação

&emsp; Para testar o novo modelo e as identificações das fissuras, o script `test_extraction.py` permite aplicar o modelo nas imagens de teste, seguindo o novo pipeline descrito abaixo:

1. Input das imagens;
2. Geração de máscaras;
3. Identificação de fissuras dentro de cada imagem de input;
4. Corte (crop) das fisuras identificadas dentro de cada imagem;
5. Classificação de cada corte de fissura identificado;
6. Segmentação das fissuras por *crop*;
7. Sobreposição das segmentações na imagem original.

* Neste novo *pipeline*, utiliza-se o novo *script* de geração de máscaras.

&emsp; Desta forma, utilizamos o modelo de classificação previamente apresentado na *Sprint 2* para a classificação dos *crops* de fissuras, mantendo métricas excelentes:

## Desempenho:

* **Acurácia:** 95%
* **Precision (thermal):** 0.91 | **Recall:** 1.00 | **F1-score:** 0.95
* **Precision (retraction):** 1.00 | **Recall:** 0.90 | **F1-score:** 0.95
* **Macro avg / Weighted avg F1-score:** 0.95

&emsp; Esse fluxo permite reaproveitamento modular, validação visual e explicabilidade.

---

## Como rodar o modelo

## Conclusão

