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

&emsp; Na Sprint 4, foi realizada uma reformulação completa do script responsável pela geração de máscaras, com o objetivo de melhorar a precisão da segmentação das fissuras, base fundamental para o bom desempenho do modelo YOLOv8-seg.

### Versão Anterior (Sprint 3)

&emsp; Na versão anterior, o fluxo de geração de máscara era mais simples e menos robusto. Ele envolvia:

1. Conversão para escala de cinza;
2. Suavização com filtro bilateral;
3. Realce com CLAHE;
4. Aplicação do filtro Sato (adequado para estruturas lineares);
5. Binarização com threshold fixo;
6. Operações morfológicas básicas (fechamento);
7. Remoção de ruído por tamanho mínimo de região.

&emsp; Apesar de funcional, esse fluxo era muito sensível à variação de iluminação e contraste entre as imagens, gerando máscaras que muitas vezes falhavam em capturar fissuras muito finas ou com ruído.

### Nova Versão (Sprint 4)

&emsp; A nova versão introduzida trouxe um fluxo mais inteligente, adaptável e modular, com melhorias em vários pontos-chave:

#### Pré-processamento aprimorado

* Aplicação de CLAHE com parâmetros otimizados;
* Filtros múltiplos de detecção de estruturas finas: além do Sato, agora são utilizados também Frangi e Meijering, cada um com diferentes sensibilidades a estruturas tubulares;
* Combinação ponderada dos filtros para gerar uma imagem realçada mais robusta.

#### Threshold adaptativo

* Análise estatística da imagem para definir thresholds baseados em média e desvio padrão dos pixels, garantindo maior adaptação a diferentes condições de imagem;
* Execução de múltiplas tentativas com valores ajustados automaticamente.

#### Pós-processamento avançado

* Uso de operações morfológicas combinadas (fechamento, abertura e filtro mediano);
* Nova função de remoção de ruído baseada em propriedades geométricas, como:

  * Área mínima,
  * Aspect Ratio (fissuras são alongadas),
  * Solidez e extent da região.

#### Avaliação de qualidade da máscara

* Nova métrica de score avalia se a máscara gerada é adequada com base em:

  * Proporção da imagem coberta;
  * Tamanho médio dos componentes conectados;
  * Penalização automática por ruído excessivo ou ausência de detecção.

#### Fallback conservador

* Se todas as tentativas falharem, é ativada uma abordagem conservadora, que:

  * Usa apenas o filtro Sato;
  * Adota thresholds e kernels morfológicos mais agressivos;
  * Aplica remoção mais rigorosa de ruídos pequenos ou redondos.

### Resultados

&emsp; Com o novo script, houve grande melhora na fidelidade da segmentação, com máscaras mais precisas, contínuas e menos ruidosas. Isso impactou diretamente na qualidade das labels geradas e no desempenho final do modelo de segmentação. Além disso, o novo script é mais resiliente a variações nas imagens, garantindo consistência no treinamento e na inferência.


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


Antes de rodar o script predict_yolo_seg.py, certifique-se de:

Ter o Python instalado (versão 3.8 ou superior recomendada).

(Opcional) Ative o ambiente virtual, se estiver utilizando:

```
.\venv\Scripts\activate
```

Ter todas as dependências instaladas:

```
pip install -r requirements.txt
```
(estar dentro do diretório do modelo)

Ter o modelo treinado salvo no caminho definido dentro do script (exemplo: ../runs/segmentation_model/weights/best.pt).

Ter as imagens de teste disponíveis na pasta definida dentro do script (exemplo: ../yolo/images/test).

Estrutura de Diretórios Esperada
```
src/
└── modelos/
    └── modelo_segmentacao/
        └── scripts/
            ├── predict_yolo_seg.py
            ├── train_yolo_seg.py
            ├── mask_func.py
            ├── label_gen.py
            ├── yolov8n-seg.pt
            └── yolo/
```

Como Executar o Script
Abra o PowerShell ou terminal.

Navegue até o diretório onde está o script:

```
cd .\src\modelos\modelo_segmentacao\scripts
```

Execute o script de predição:

```
python predict_yolo_seg.py
```

Resultado

As imagens com as segmentações aplicadas serão salvas na pasta definida no script:

```
../yolo/output
```

As classes detectadas são anotadas diretamente nas imagens, e as segmentações são desenhadas na imagem com contornos.

***Observações***

- O script ignora arquivos que não sejam .jpg, .jpeg ou .png.
- Se não houver detecção em uma imagem, uma mensagem será exibida no terminal.
- É possível alterar o limiar de confiança, o modelo e os caminhos diretamente no script.



## Conclusão

&emsp; As melhorias implementadas no script de geração de máscaras e no fluxo de segmentação elevaram a precisão e a verossimilhança do modelo YOLOv8-seg. Com uma abordagem mais adaptativa e modular, o novo pipeline oferece segmentações mais confiáveis, maior fidelidade na geração de labels e um desempenho consistente mesmo em imagens com diferentes condições visuais. Esses avanços consolidam a base para futuras otimizações e aplicações práticas do modelo.