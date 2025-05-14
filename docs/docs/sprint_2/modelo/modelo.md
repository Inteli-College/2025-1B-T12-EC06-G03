---
title: Modelo de Detecção e Classificação de Fissuras
sidebar_position: 2
---

# Modelo de Detecção e Classificação de Fissuras

##### Modelo de Detecção e Classificação de Fissuras

## Introdução

Este documento descreve o funcionamento de um sistema automatizado para **detecção e classificação de fissuras** em imagens, com foco em aplicações na engenharia civil e inspeção de estruturas. O sistema utiliza dois modelos de aprendizado profundo:

* Um modelo **YOLO (You Only Look Once)** para **detecção da fissura** na imagem
* Um modelo **CNN (Convolutional Neural Network)** para **classificação do tipo da fissura**, entre *térmica* e *de retração*

A abordagem prioriza **especialização modular**, permitindo maior controle e flexibilidade no pipeline.

