---
title: Escopo da Sprint 3
sidebar_position: 0
---

# Escopo da Sprint 3

## Introdução

A Sprint 3 do projeto Athena teve como objetivo consolidar os primeiros módulos funcionais do sistema completo. Nesta fase, a equipe concentrou esforços em conectar os diferentes subsistemas da solução — incluindo o front-end, o back-end da aplicação web e o backend dos modelos de inteligência artificial. Também foram realizadas melhorias na usabilidade da interface, adição de autenticação e autorização de usuários, bem como avanços significativos na implementação dos modelos de classificação e segmentação de fissuras.

Essa sprint marca o início da integração plena entre as partes técnicas da solução, preparando o terreno para testes e simulações de uso real nos ciclos seguintes.

## Funcionalidades Implementadas

### Atualizações no Front-End
- **Reestruturação Visual**: Melhorias na organização das telas e navegação da interface web.
- **Componentes Novos**: Criação de novos componentes para visualização de imagens, controle de status e navegação.
- **Feedback Visual**: Inclusão de indicadores de carregamento e mensagens de sucesso/erro ao interagir com a API.
- **Design Consistente**: Adequação ao guia visual e identidade do projeto.

### Integração com o Back-End
- **Chamadas REST**: Conexão entre o front-end e os endpoints da API usando `fetch` e `axios`.
- **Upload de Imagens**: Implementação da funcionalidade de upload de imagens com redirecionamento automático para análise.
- **Comunicação com o Modelo**: Configuração da lógica que envia imagens para o back-end Python e exibe os resultados na tela.

### Back-End da Aplicação Web
- **Servidor Express/Flask**: Estruturação de uma API para gerenciar autenticação, upload de imagens e interação com o modelo.
- **Autenticação e Autorização**: Adição de login e controle de permissões com tokens de sessão.
- **Gerenciamento de Sessões**: Sistema de login com redirecionamento de rotas baseado em perfil (ex: admin, técnico do IPT).

### Backend do Modelo Preditivo
- **Servidor Python para Modelos**: Criação de um back-end leve com Flask para receber e processar requisições dos modelos.
- **Pipeline de Processamento**: Recebimento da imagem, pré-processamento, inferência com o modelo e retorno da classificação ao front-end.
- **Rotas Específicas**: Criação de rotas distintas para modelos de classificação e segmentação.

### Modelo de Classificação de Fissuras
- **Treinamento Inicial**: Treinamento de um classificador com base em imagens reais de fissuras (retração e térmica).
- **Validação Cruzada**: Avaliação do desempenho com métricas como acurácia, F1-score e matriz de confusão.
- **Exportação do Modelo**: Serialização do classificador (.pkl) e integração ao servidor Flask.

### Primeira Versão do Modelo de Segmentação de Fissuras
- **Pré-processamento**: Definição da pipeline de input para segmentação de regiões da imagem com presença de fissuras.
- **Inferência Pixel a Pixel**: Estruturação de máscaras sobrepostas às imagens originais.
- **Salvamento de Resultados**: Armazenamento temporário de arquivos segmentados para visualização futura.

### Interface do Modelo
- **Interface de Análise**: Tela dedicada à exibição dos resultados da classificação e da segmentação.
- **Visualização com Sobreposição**: Visualização da imagem original com a máscara da segmentação por cima.
- **Feedback Explicativo**: Mensagens interpretativas para ajudar o usuário a entender o tipo de fissura detectado.

## Conclusão

A Sprint 3 representou um avanço técnico importante para o projeto Athena, com a primeira integração real entre o front-end, o back-end e os modelos de IA. A partir dessa base funcional integrada, o projeto está mais próximo de se tornar uma ferramenta completa de apoio à inspeção de fissuras, capaz de gerar relatórios confiáveis e análises automatizadas para os profissionais do IPT. A próxima sprint focará em testes de robustez, melhorias na interface e início da geração automatizada de relatórios.
