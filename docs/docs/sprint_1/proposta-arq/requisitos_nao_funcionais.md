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

## RNF3 – Precisão na Detecção de Fissuras

**Descrição**:  
&emsp;O sistema Athena deve detectar fissuras em revestimentos de argamassa com acurácia mínima de 90%, mantendo taxa de falsos positivos abaixo de 5% e falsos negativos abaixo de 7%, em condições normais de iluminação (300 a 1000 lux).

**Justificativa**:  
&emsp;Uma taxa elevada de falsos positivos comprometeria a eficiência das inspeções, enquanto falsos negativos podem permitir que danos estruturais passem despercebidos. Altos padrões de precisão são indispensáveis para aplicações de engenharia.

**Métrica**:  
- Acurácia ≥ 90%;
- Taxa de falsos positivos < 5%;
- Taxa de falsos negativos < 7%;
- Desvio padrão da acurácia entre condições < 3%.

**Método de Teste Aprofundado**:
- **Ambiente**:
    - Setup de iluminação controlada (luxímetro);
    - Câmera de alta resolução (mín. 12MP).
- **Ferramentas**:
    - Python (OpenCV para detecção + Sklearn para métricas).
- **Procedimento**:
    1. Capturar 100 imagens nas condições:
        - Iluminação 300 lux (baixa);
        - 600 lux (média);
        - 1000 lux (alta).
    2. Processar todas no sistema Athena.
    3. Marcar VP, FP, FN, VN manualmente (comparado com rótulo ouro).
    4. Calcular Acurácia, FP rate, FN rate para cada condição.
- **Critério de Aceitação**:
- Acurácia ≥ 90% em todas condições;
- FP < 5%, FN < 7%;
- Desvio padrão da acurácia entre condições < 3%.
---

## RNF4 – Autonomia da Bateria do Drone

**Descrição**:  
&emsp;O drone utilizado no projeto deve operar continuamente por pelo menos 15 minutos em voo ativo, com câmera e sensores de inspeção funcionando, sob carga de trabalho padrão e sem necessidade de troca ou recarga da bateria.

**Justificativa**:  
&emsp;Em ambientes de difícil acesso, a interrupção frequente para troca de bateria pode inviabilizar inspeções eficientes. Garantir no mínimo 15 minutos permite a cobertura adequada de grandes áreas sem comprometer a logística da operação.

**Métrica**:  
- Duração mínima de voo contínuo ≥ 15 minutos, atingindo 20% de carga residual.

**Método de Teste Aprofundado**:
- **Ambiente**:
    - Campo aberto, temperatura entre 14°C e 40°C;
    - Sem vento forte (>15km/h).
- **Ferramentas**:
    - Aplicativo de monitoramento de bateria do drone (DJI GO 4 ou equivalente).
- **Procedimento**:
    1. Realizar voo padrão simulando inspeção (movimentação lateral e vertical).
    2. Iniciar cronômetro no momento da decolagem.
    3. Registrar o tempo até a bateria atingir 20% de carga.
    4. Repetir 5 vezes em dias diferentes.
- **Critério de Aceitação**:
    - 5 voos com tempo ≥ 15 minutos;
    - Se falhar, investigar condições externas (vento, temperatura) antes de reprovar.
---

## RNF5 – Latência do Comando de Captura de Imagem

**Descrição**:  
&emsp;O intervalo de tempo entre o clique do operador no botão de captura e o recebimento da confirmação de armazenamento no servidor deve ser de no máximo 2 segundos em 95% das execuções realizadas sob rede 4G/LTE típica.

**Justificativa**:  
&emsp;Reduzir a latência de captura é crucial para alinhar o movimento do drone à tomada de imagens de alta qualidade, evitando tempo ocioso e otimizando o consumo energético durante as inspeções.

**Métrica**:  
- L95 (95º percentil de latência) ≤ 2 segundos;
- Média de latência ≤ 1,5 segundos;
- Latência máxima ≤ 3 segundos.

**Método de Teste Aprofundado**:
- **Ambiente**:
    - Interface web do sistema conectada via modem 4G Cat 4.
- **Ferramentas**:
    - Cypress para automação de cliques;
    - Logger de timestamps com precisão de milissegundos.
- **Procedimento**:
    1. Configurar script Cypress para enviar comando de captura a cada 3s.
    2. Realizar 3 lotes de 100 capturas cada (300 capturas no total).
    3. Registrar para cada captura:
        - T₀ = tempo do clique;
        - T₂ = tempo do recebimento do ack no servidor.
    4. Calcular para cada lote:
        - Lᵢ = T₂ - T₀
        - L₉₅ (percentil 95);
        - Média;
        - Máximo.
- **Critério de Aceitação**:
- L95 ≤ 2s em todos os lotes;
- Média ≤ 1,5s;
- Máximo ≤ 3s.
---

