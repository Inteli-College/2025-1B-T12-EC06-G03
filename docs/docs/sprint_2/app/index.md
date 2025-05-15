---
title: Visão Geral do App Athenas
sidebar_position: 0
---

# Visão Geral do App Athenas

## Introdução

O App Athenas é uma aplicação móvel desenvolvida em Flutter para o controle remoto de drones. Este aplicativo foi projetado para fornecer uma interface de usuário intuitiva e responsiva que permite aos operadores controlar drones em tempo real, receber feedback visual através de streaming de vídeo e monitorar telemetria.

## Características Principais

- **Interface Intuitiva**: Design minimalista e funcional em modo paisagem otimizado para tablets e smartphones
- **Controles de Joystick Duplos**: Para movimentação precisa (direção e altitude)
- **Streaming de Vídeo em Tempo Real**: Visualização do feed da câmera do drone com baixa latência
- **Telemetria**: Monitoramento de dados como bateria, altitude, velocidade e qualidade do sinal
- **Comandos Rápidos**: Botões de decolagem, pouso, retorno automático e parada de emergência
- **Configurações Personalizáveis**: Ajustes de IP, porta e outras configurações de conectividade
- **Modo Noturno**: Interface escura para melhorar a visibilidade em diferentes condições de iluminação

## Arquitetura do App

O aplicativo Athenas segue o padrão de arquitetura BLoC (Business Logic Component) para gerenciamento de estado, separando claramente:

- **UI (Interface do Usuário)**: Widgets Flutter reativos
- **BLoC**: Componentes de lógica de negócios que processam eventos e emitem estados
- **Serviços**: Camada de comunicação com o servidor do drone
- **Modelos**: Estruturas de dados para comandos e respostas

## Tecnologias Utilizadas

- **Flutter**: Framework para desenvolvimento multiplataforma
- **Dart**: Linguagem de programação
- **Flutter BLoC**: Gerenciamento de estado
- **Socket.io Client**: Comunicação em tempo real com o servidor
- **Dio**: Cliente HTTP para chamadas de API REST

## Requisitos do Sistema

- **Sistemas Operacionais**: iOS 12+ ou Android 8.0+
- **Orientação**: Modo paisagem obrigatório
- **Conectividade**: Wi-Fi para conexão com o servidor do drone
- **Armazenamento**: Mínimo de 100MB para instalação do aplicativo

## Navegação na Documentação

Para informações detalhadas sobre componentes específicos do aplicativo, consulte:

- [Sistema de Controle](controle): Detalhes sobre o sistema de controle do drone
- [Configuração do App](config): Instruções para configurar o aplicativo
- [Interface do Usuário](interface): Informações sobre a UI do aplicativo
- [Arquitetura Técnica](arquitetura): Detalhes da arquitetura de software
- [Sistema de Comunicação](comunicacao): Detalhes sobre a comunicação com o servidor