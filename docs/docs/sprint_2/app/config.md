---
title: Configuração do App
sidebar_position: 2
---

# Configuração do App

## Configurações de Conexão

O aplicativo Athenas precisa se conectar ao servidor de controle do drone. Esta seção detalha as configurações necessárias para estabelecer essa conexão.

### Configurações do Servidor

As configurações padrão do servidor estão definidas no modelo `ServerConfig`:

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| Host      | 10.32.0.11   | Endereço IP do servidor de controle do drone |
| Porta     | 5000         | Porta de comunicação do servidor |
| Caminho de Salvamento | Automático | Diretório onde serão salvos vídeos e fotos capturados |

### Tela de Configuração do Servidor

A tela de configuração (`ServerConfigScreen`) permite ao usuário:

1. Alterar o endereço IP do servidor
2. Modificar a porta de comunicação
3. Escolher o diretório para salvar mídias capturadas
4. Testar a conexão antes de aplicar as configurações
5. Salvar as configurações para uso futuro

## Preferências de Exibição

### Modo Escuro

O aplicativo utiliza o tema escuro por padrão para:
- Melhorar a visibilidade em ambientes externos
- Reduzir o consumo de bateria em telas OLED/AMOLED
- Minimizar distrações visuais durante operações críticas

### Orientação da Tela

O aplicativo Athenas foi projetado exclusivamente para uso em orientação paisagem (landscape), sendo esta configuração forçada através do código:

```dart
await SystemChrome.setPreferredOrientations([
  DeviceOrientation.landscapeLeft,
  DeviceOrientation.landscapeRight,
]);
```

### Modo Tela Cheia

Para maximizar a área de visualização, o aplicativo opera em modo de tela cheia:

```dart
await SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
```

## Configurações de Controle

### Sensibilidade dos Joysticks

Os joysticks virtuais podem ser calibrados quanto à sua sensibilidade:

- **Joystick Direcional**: Controla movimentos horizontais (frente/trás, esquerda/direita)
- **Joystick de Altitude**: Controla movimentos verticais (subida/descida) e rotação (yaw)

As configurações de sensibilidade determinam quão rápido o drone responde aos comandos do joystick, permitindo ajustes para diferentes níveis de habilidade do piloto.

### Limites de Velocidade

É possível configurar limites máximos de velocidade para garantir operações seguras:

- **Velocidade Horizontal**: 0-100% da velocidade máxima do drone
- **Velocidade Vertical**: 0-100% da velocidade máxima de subida/descida
- **Velocidade de Rotação**: 0-100% da velocidade máxima de rotação

