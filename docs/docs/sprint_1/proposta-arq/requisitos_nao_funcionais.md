---
title: Requisitos Não Funcionais
sidebar_position: 3
---

# Requisitos Não Funcionais (RNFs)  

##### Requisitos não funcionais do projeto

&emsp;Requisitos não funcionais definem critérios de qualidade, desempenho, usabilidade e confiabilidade que o sistema deve atender, assegurando seu funcionamento adequado em ambiente real. Assim, as especificações aqui descritas abrangem desde a performance de transmissão de vídeo até a eficiência do processamento de dados e a confiabilidade dos mecanismos de captura e análise de imagens do nosso projeto em desenvolvimento.
O atendimento a esses requisitos é fundamental para a entrega de uma solução robusta, eficiente e capaz de operar em condições práticas desafiadoras, respeitando padrões técnicos exigidos para aplicações de engenharia e manutenção predial.

## RNF1 – Qualidade da Transmissão de Vídeo

**Descrição**:  
&emsp;O sistema de transmissão de vídeo do drone para a plataforma de análise deve garantir uma comunicação em tempo real, assegurando uma taxa mínima de 10 frames por segundo (FPS) e latência de transmissão inferior a 500 milissegundos, mesmo sob variações moderadas de rede. A estabilidade da transmissão é essencial para permitir que operadores capturem imagens no momento correto e acompanhem as inspeções sem atrasos perceptíveis ou travamentos críticos.

**Justificativa**:  
&emsp;Em inspeções de campo, decisões precisam ser tomadas rapidamente com base no fluxo de vídeo. Quedas de FPS ou picos de latência podem gerar imagens desatualizadas, perda de detalhes em fissuras ou até falhas operacionais do drone. Assim, a qualidade da transmissão é um pilar da eficácia da solução.

**Métrica**:  
- FPS ≥ 10 em sessões contínuas de 10 minutos;
- Latência ≤ 500 ms em 95% das medições;
- Oscilação de FPS < 15% durante variações de sinal.

**Método de Teste Aprofundado**:
- **Ambiente**:
    - Drone em operação real conectado via Wi-Fi 5GHz ou 4G com sinal típico (>10 Mbps).
    - Plataforma de visualização de vídeo ativa no notebook de controle.
- **Ferramentas**:
    - OpenCV para medir FPS recebido em tempo real;
    - Wireshark ou tcpdump para medir latência de pacotes.
- **Procedimento**:
    1. Configurar o ambiente em local de teste externo ou interno com boa cobertura de rede.
    2. Operar o drone transmitindo vídeo por 10 minutos contínuos em três sessões distintas:
        - Sem interferência (ideal),
        - Com perda simulada de pacotes (~5%),
        - Com oscilação de sinal (mudança de 4G para Wi-Fi).
    3. Registrar o FPS e latência em cada sessão.
    4. Calcular FPS médio, variação de FPS e latência média e máxima.
- **Critério de Aceitação**:
    - FPS médio ≥ 10;
    - 95% das medições de latência ≤ 500 ms;
    - Oscilação de FPS (desvio padrão / média) < 15%.
---

## RNF2 – Acurácia do Modelo de Classificação de Fissuras

**Descrição**:  
&emsp;O modelo de Machine Learning desenvolvido para classificar fissuras deve apresentar acurácia mínima de 70%, com consistência entre execuções validada por variação máxima de 5%. A avaliação será feita com base em imagens reais, não utilizadas no treinamento.

**Justificativa**:  
&emsp;Modelos instáveis comprometem a confiança no sistema de inspeção. A estabilidade da acurácia demonstra que o modelo é robusto e confiável em ambientes reais de operação, respeitando os padrões de engenharia civil.

**Métrica**:  
- Acurácia média ≥ 70% nas execuções;
- Diferença entre a maior e menor acurácia ≤ 5%.

**Método de Teste Aprofundado**:
- **Ambiente**:
    - Conjunto de teste com pelo menos 500 imagens reais rotuladas por especialistas.
- **Ferramentas**:
    - Sklearn (cross_val_score), scripts Python para validação.
- **Procedimento**:
    1. Dividir o dataset em 5 partes balanceadas (stratified split).
    2. Treinar e validar o modelo 5 vezes, cada vez usando uma parte como teste.
    3. Registrar a acurácia obtida em cada rodada.
- **Critério de Aceitação**:
    - Média das 5 acurácias ≥ 70%;
    - Diferença entre a maior e a menor acurácia ≤ 5%;
    - Se não atender, revisar hiperparâmetros ou conjunto de treino.
---

