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

### Desempenho:

* **Precisão média (mAP\@0.5):** 0.709
* **Revocação média:** 0.789
* **mAP\@0.5:0.95:** 0.400
* **Velocidade de inferência:** \~95ms por imagem

Esses resultados indicam uma boa capacidade do modelo em detectar fissuras com alta confiança, mesmo em cenários variados.

---

## Modelo CNN - Classificação da Fissura

A CNN é responsável por **analisar visualmente o recorte da fissura** e classificá-la como:

* **Térmica:** geralmente mais grossa, retilínea
* **Retração:** fina, ramificada ou com bordas irregulares

### Estrutura do modelo:

* 2 camadas convolucionais com ReLU e MaxPooling
* Camadas densas com dropout para evitar overfitting
* Camada final com duas saídas para classificação binária
* Função de perda: `CrossEntropyLoss`
* Otimizador: `Adam`

### Desenvolvimento:

O modelo foi treinado com **180 imagens rotuladas** divididas em duas subpastas: `thermal` e `retraction`. As imagens foram recortes manuais da região da fissura (obtidas após anotação do YOLO), organizadas em `train`, `val` e `test`, com 10 imagens para validação e 10 para teste. As imagens foram transformadas para tons de cinza, redimensionadas para 128x128 pixels e normalizadas.

O modelo foi salvo com seu respectivo `class_to_idx.json`, garantindo correspondência correta entre índices e nomes das classes na inferência.

### Desempenho:

* **Acurácia:** 95%
* **Precision (thermal):** 0.91 | **Recall:** 1.00 | **F1-score:** 0.95
* **Precision (retraction):** 1.00 | **Recall:** 0.90 | **F1-score:** 0.95
* **Macro avg / Weighted avg F1-score:** 0.95

Esses números indicam que o modelo consegue generalizar bem para ambas as classes, mesmo com base de teste pequena.

---

## Por que dois modelos? 

A escolha por utilizar **dois modelos distintos e especializados** em vez de um único modelo multitarefa se baseia em princípios técnicos de separação de responsabilidades, complexidade de tarefa e especialização de arquitetura:

1. **Tarefas distintas exigem capacidades diferentes:**

   * A detecção de objetos (YOLO) requer identificar com precisão *onde* algo está na imagem.
   * A classificação (CNN) requer avaliar *o que é* aquilo, com foco em características mais finas de forma e textura.

2. **YOLO como detector binário:**

   * O YOLO foi treinado como um detector binário: detecta se há uma fissura, sem se preocupar com o tipo.
   * Treiná-lo para detecção + classificação multiclasse exigiria um dataset muito maior, com anotações de bounding boxes por tipo de fissura — o que aumentaria significativamente a complexidade da curadoria.

3. **CNN como classificador especializado:**

   * A CNN é alimentada apenas com a área recortada da imagem (a fissura detectada) e foca apenas na diferenciação entre *térmica* e *retração*.
   * Isso reduz o ruído da imagem original e permite que o classificador aprenda padrões visuais com mais precisão.

4. **Vantagens do desacoplamento:**

   * Permite reusar o YOLO com outros classificadores ou adaptar a CNN para novas classes de fissura.
   * Otimizações podem ser feitas separadamente: o YOLO pode ser treinado com imagens em diversos contextos, enquanto a CNN pode ser refinada apenas com melhorias de recorte e balanceamento de classes.

5. **Evita sobrecarga no YOLO e viés por contexto:**

   * O YOLO pode sofrer com viés se tentar classificar tipos de fissura baseando-se em informações contextuais da imagem (ex: iluminação, fundo).
   * Ao separar as tarefas, a CNN foca exclusivamente na análise visual da fissura em si, evitando interferência do cenário.

6. **Facilidade de anotação:**

   * Anotar apenas a presença de fissura (como no YOLO binário) é mais simples e rápido.
   * Classificar recortes já feitos (como no caso da CNN) é mais intuitivo e exige menos esforço de curadoria do que rotular bounding boxes multiclasse.

7. **Resultados comprovam a eficácia:**

   * O YOLO obteve mAP\@0.5 de 0.709, suficiente para recortar com precisão a fissura.
   * A CNN, com recortes já centrados na fissura, alcançou 95% de acurácia na classificação entre os dois tipos.

Essa abordagem modular representa uma boa prática de engenharia de sistemas de visão computacional, pois equilibra desempenho, flexibilidade e interpretabilidade.

---

## Integração dos Modelos

Essa conexão entre os modelos foi validada por meio de um script integrado chamado `detect_and_classify.py`, que permite realizar a inferência completa a partir de uma única imagem de entrada. Ele executa o pipeline completo:

1. Recebe uma imagem original.
2. Aplica o pré-processamento para o YOLO.
3. O YOLO retorna as coordenadas da fissura com maior confiança.
4. A área detectada é recortada da imagem original (sem filtros do YOLO).
5. O recorte é transformado e passado ao modelo CNN.
6. O resultado é impresso na imagem original com texto e caixa.
7. As saídas são salvas como:

* `resultado_final.png`: imagem com bounding box e rótulo da classificação da fissura
* `resultado_yolo_bruto.png`: imagem com apenas o box do YOLO e nível de confiança da fissura selecionada

Esse fluxo permite reaproveitamento modular, validação visual e explicabilidade.

