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

---


## Modelo YOLO - Detecção de Fissura

O modelo YOLOv8 é responsável por identificar a presença de **fissuras visíveis** em uma imagem. Ele foi treinado com imagens anotadas com caixas delimitadoras (bounding boxes) marcando a localização da fissura.

### Desenvolvimento:

Antes de serem usadas no treinamento, as imagens passaram por um processo de melhoria visual utilizando o script `preprocess_images.py`. Esse script aplicou técnicas como CLAHE (equalização adaptativa), blur e realce por nitidez. As imagens processadas foram então anotadas manualmente com o aplicativo **LabelImg**, que gerou os arquivos `.txt` contendo as caixas delimitadoras no formato esperado pelo YOLO. Cada linha do `.txt` descreve a classe (sempre "fissura") e as coordenadas normalizadas da bounding box.

O conjunto de dados utilizado para o treino do YOLO continha **aproximadamente 180 imagens**, além de **10 para validação** e **10 para teste**. As imagens incluíam fissuras térmicas e de retração misturadas, uma vez que o objetivo do YOLO era apenas detectar a presença de uma fissura, independentemente do tipo.

O treinamento foi realizado com a versão YOLOv8n por 30 épocas, utilizando tamanho de imagem 640x640 e batch size de 8. O script de treinamento usou o comando `.train(data='fissure.yaml', ...)`, com os resultados sendo salvos na pasta `runs/detect/fissura-detector`.

### Funcionamento:

1. A imagem é processada (CLAHE, equalização e realce) e passada ao modelo YOLO.
2. O modelo retorna uma ou mais detecções com coordenadas (x1, y1, x2, y2) e uma **pontuação de confiança**.
3. A **detecção com maior confiança** é selecionada - já que o objetivo é selecionar uma fissura por imagem.
4. A região correspondente é recortada da imagem original.

