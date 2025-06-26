# Documentação Técnica sobre as Mudanças no Modelo de Segmentação de Fissuras

Este documento descreve as alterações e melhorias implementadas no modelo de **segmentação de fissuras térmicas e de retração** utilizando **YOLO** e **Streamlit**. As mudanças abordam desde otimizações na carga e manipulação do modelo até ajustes na interface de usuário, visando melhorar tanto a performance quanto a usabilidade da aplicação.

## 1. **Otimização do Carregamento do Modelo com Cache**

Uma das alterações mais significativas foi a **otimização do carregamento do modelo YOLO**. A função `load_model`, que antes realizava a inicialização do modelo de forma direta, foi aprimorada com o decorador `@st.cache_resource`. Esse cache permite que o modelo seja carregado apenas uma vez, armazenando a instância em memória e evitando carregamentos repetidos. Isso resulta em uma inicialização mais rápida, especialmente quando o modelo precisa ser carregado após interações subsequentes.

**Alteração Técnica:**
- Uso do `@st.cache_resource` para armazenar o modelo em cache, otimizando o tempo de inicialização nas interações subsequentes.

## 2. **Gerenciamento e Manipulação de Uploads de Imagens**

Uma das falhas da versão anterior era o tratamento inadequado de uploads em sessões consecutivas. Agora, a aplicação realiza uma **limpeza prévia das pastas de upload e output** (`UPLOAD_FOLDER` e `OUTPUT_FOLDER`) sempre que novas imagens são carregadas, utilizando as funções `shutil.rmtree()` e `os.makedirs()`. Isso garante que não haja sobrecarga de arquivos de uploads anteriores, mantendo a integridade do fluxo de predição.

**Alteração Técnica:**
- Implementação de **limpeza automática das pastas de upload e output** antes de novos uploads com `shutil.rmtree()` e recriação das pastas com `os.makedirs()`.

## 3. **Visualização das Imagens Carregadas e Organizadas**

A visualização das imagens carregadas foi aprimorada para ser **mais responsiva** e intuitiva. As imagens são exibidas em **colunas dinâmicas** (`st.columns(3)`), com um limite de 3 imagens por linha, proporcionando uma interface mais organizada. Essa alteração visa melhorar a experiência visual do usuário, permitindo uma visualização clara das imagens carregadas antes da predição.

**Alteração Técnica:**
- Utilização de **colunas dinâmicas** para exibição das imagens carregadas, distribuindo as imagens de forma eficiente com `st.columns(3)`.

## 4. **Aprimoramento da Função de Predição**

A função `run_prediction` passou por uma reformulação significativa para proporcionar **resultados de segmentação mais precisos e visualmente claros**. Agora, o modelo gera uma **máscara binária combinada** para a segmentação de fissuras, utilizando operações de bitwise com o OpenCV para sobrepor as áreas segmentadas sobre as imagens originais. As máscaras segmentadas são ajustadas com base no tamanho das imagens originais, garantindo um alinhamento preciso.

Além disso, a função agora coleta e exibe as **contagens de classes** detectadas em cada imagem. A legenda gerada para cada predição contém informações sobre o tipo de fissura e a confiabilidade da predição.

**Alteração Técnica:**
- **Máscaras binárias combinadas**: Utilização de `cv2.resize()` para ajustar as máscaras e `cv2.bitwise_or()` para combiná-las.
- **Sobreposição das máscaras**: Uso de `cv2.addWeighted()` para aplicar transparência e destacar as fissuras detectadas.
- **Exibição de contagens de classes**: Contagem das classes detectadas e exibição de uma legenda informativa com o nome da classe mais predita.

## 5. **Visualização dos Resultados da Segmentação**

A exibição dos resultados de segmentação foi aprimorada para tornar as predições mais claras e acessíveis. Agora, cada imagem predita é exibida com a **máscara de segmentação sobreposta**, usando um filtro de cor específico para destacar as fissuras detectadas. A legenda inclui o nome da classe predita (Térmica ou Retração) e o nome da imagem. O layout da página foi ajustado para suportar até **três imagens por linha**, organizando os resultados de forma eficiente.

**Alteração Técnica:**
- **Exibição da imagem sobreposta**: Implementação de `cv2.addWeighted()` para sobrepor a máscara de segmentação sobre a imagem original.
- **Layout responsivo**: Uso de `st.columns(3)` para exibir os resultados em um formato de 3 colunas, otimizando a visualização.

## 6. **Melhoria no Feedback Visual e Interatividade**

O feedback visual foi melhorado para aumentar a clareza da interface. Durante o processamento da predição, um **spinner de carregamento** é exibido para indicar que o sistema está processando a imagem, e, após a predição, uma **mensagem de sucesso** é exibida. Essas alterações visam melhorar a experiência do usuário ao fornecer informações claras sobre o andamento do processamento.

**Alteração Técnica:**
- Implementação de **feedback visual interativo** com `st.spinner()` para mostrar que o processo está em andamento e `st.success()` para indicar que o processo foi concluído.

## 7. **Estilização e Ajustes na Interface de Usuário**

A interface foi **estilizada para melhorar a legibilidade e a apresentação**. Foram feitas alterações no CSS, como o aumento no tamanho da fonte para melhorar a leitura e o ajuste nos estilos de título e texto. Essas modificações visam proporcionar uma experiência mais confortável ao usuário ao interagir com a plataforma.

**Alteração Técnica:**
- **Customização de CSS** utilizando `st.markdown()` para aumentar o tamanho da fonte e ajustar o espaçamento de elementos, como títulos e seções, visando melhorar a legibilidade.

## 8. **Reorganização do Fluxo de Trabalho**

A reorganização do fluxo de trabalho permite que o upload e a predição ocorram de maneira mais intuitiva. Antes da execução da predição, o upload das imagens é realizado de forma clara, e o botão para rodar a predição só se torna disponível após as imagens serem carregadas. Isso melhora a estrutura do processo, proporcionando uma navegação mais fluida.

**Alteração Técnica:**
- **Estrutura de etapas claras**: Separação entre o upload das imagens e a execução da predição, com a ativação do botão de predição somente após o upload ser completado.

## 9. **Limitações e Desafios na Integração com o Frontend**

Embora o modelo de segmentação tenha sido otimizado e melhorado no backend, **não foi possível concluir a integração com o frontend a tempo**. Isso se deve principalmente à **diferença substancial entre a arquitetura do modelo de classificação e o modelo de segmentação**. O modelo de classificação utilizado anteriormente tinha um formato e um fluxo de dados mais simples, enquanto o modelo de segmentação exige um processamento mais complexo de **máscaras binárias** e **sobreposição de imagens**, o que dificultou a integração rápida com o frontend. 

Além disso, a manipulação das imagens segmentadas e o envio de dados para o frontend precisariam ser ajustados para garantir a compatibilidade com as expectativas da interface de usuário, o que demandaria mais tempo de desenvolvimento. Portanto, a integração do modelo de segmentação com o frontend será realizada em uma próxima etapa, após o ajuste dessas discrepâncias.

---

### Conclusão

As modificações implementadas visam **otimizar a performance**, melhorar a **experiência de usuário** e garantir que os resultados da segmentação de fissuras sejam apresentados de forma clara e precisa. As alterações incluem desde a otimização do carregamento do modelo com cache até a melhoria na visualização das predições e na interação com o usuário. As melhorias técnicas incluem:

- **Cache de carregamento do modelo** para reduzir o tempo de inicialização.
- **Limpeza automática das pastas de upload** para evitar sobrecarga de dados.
- **Melhoria na visualização e exibição dos resultados** com máscaras combinadas e legendas informativas.
- **Aprimoramento no feedback interativo** com mensagens de carregamento e sucesso.
- **Reorganização do fluxo de trabalho**, proporcionando uma experiência mais intuitiva.

Essas melhorias fazem com que o sistema esteja agora **mais eficiente, interativo e responsivo**, permitindo uma análise de segmentação de fissuras de maneira mais rápida e precisa.

Apesar da implementação bem-sucedida dessas melhorias, **a integração com o frontend não pode ser feita devido à diferença no fluxo de dados e processamento** entre o modelo de classificação e o modelo de segmentação. Esse processo poderá ser continuado em uma próxima fase do projeto, quando for efetivamente implementado.
