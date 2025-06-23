---
title: Virtual Reality (VR)
sidebar_position: 4
---

# Aplicação VR para Controle de Drone

&emsp;A aplicação VR desenvolvida permite controlar drones através de uma interface imersiva em realidade virtual, utilizando A-Frame como framework principal para criar experiências WebXR.

## Visão Geral

&emsp;O sistema VR consiste em uma aplicação web que utiliza tecnologias WebXR para criar uma interface de controle de drone em ambiente de realidade virtual e realidade aumentada. A aplicação permite visualizar o stream de vídeo do drone e executar comandos básicos através de interações VR.

### Tecnologias Utilizadas

- **A-Frame 1.6.0**: Framework principal para desenvolvimento WebXR
- **WebXR AR**: Suporte para realidade aumentada
- **Super Hands**: Sistema de interação com objetos 3D
- **Hand Tracking Controls**: Controle por rastreamento de mãos
- **Laser Controls**: Sistema de ponteiro laser para controles VR
- **Flask**: Backend Python para servir a aplicação

## Arquitetura da Aplicação

### Estrutura de Arquivos

```
src/vr-app/
├── app.py                 # Servidor Flask
├── templates/
│   └── index.html        # Interface VR principal
└── static/              # Recursos estáticos
```

### Componentes Principais

#### 1. Servidor Flask (`app.py`)

&emsp;O servidor Flask serve como backend da aplicação, fornecendo:
- Rota principal (`/`) que renderiza a interface VR
- Rota de vídeo (`/video`) para streaming (placeholder)
- Rota simplificada (`/simple`) para testes

#### 2. Interface VR (`index.html`)

A interface principal contém:
- **Cena A-Frame**: Ambiente 3D configurado para VR/AR
- **Controles VR**: Suporte para controladores e rastreamento de mãos
- **Tela de Vídeo**: Visualização do stream do drone
- **Botões Interativos**: Controles para comandos do drone
- **Componentes Personalizados**: Scripts JavaScript para funcionalidades específicas

## Funcionalidades Implementadas

### 1. Visualização de Vídeo

- Tela virtual que exibe o stream do drone
- Controles para mover, rotacionar e redimensionar a tela
- Suporte para interação via VR e desktop

### 2. Controles de Drone

Botões interativos disponíveis:
- **TAKEOFF**: Comando de decolagem
- **LAND**: Comando de pouso
- **BATTERY**: Verificação de bateria
- **FLIP**: Comando de manobra

### 3. Sistemas de Interação

- **Controles VR**: Suporte para controladores Oculus/Meta Quest
- **Rastreamento de Mãos**: Interação sem controladores
- **Controle por Teclado**: Fallback para desktop
- **Ponteiro Laser**: Sistema de seleção à distância

## Configuração e Execução

### Requisitos

- Python 3.8+
- Flask
- Navegador com suporte WebXR
- Dispositivo VR compatível (opcional)

### Instalação

```bash
cd src/vr-app
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install flask
```

### Execução

```bash
python app.py
```

A aplicação estará disponível em `http://localhost:5000`

## Componentes Personalizados

### moveable-screen

&emsp;Componente que permite interagir com a tela de vídeo:
- Movimento no espaço 3D
- Rotação
- Redimensionamento
- Indicadores visuais de interatividade

### interactive-button

&emsp;Componente para botões de comando:
- Estados visuais (normal, hover, pressionado)
- Integração com comandos de drone
- Feedback tátil e visual

### vr-debug

&emsp;Sistema de debug para desenvolvimento:
- Logging de eventos VR
- Monitoramento de interações
- Diagnóstico de problemas

## Problemas Conhecidos e Limitações

### ⚠️ Problemas de Integração com Controles VR

&emsp;Atualmente, estamos enfrentando alguns desafios na integração completa com os controles VR:

#### 1. Detecção Inconsistente de Controladores

**Problema**: Os controladores VR nem sempre são detectados corretamente, especialmente em:
- Dispositivos Meta Quest 2/3
- Navegadores diferentes (Chrome vs Edge vs Firefox)
- Primeira inicialização da sessão VR

**Sintomas**:
- Controladores não aparecem na cena
- Raycasters não funcionam
- Eventos de interação não são disparados

**Tentativas de Solução**:
```javascript
// Configuração atual dos controladores
<a-entity id="leftController" 
  hand-controls="hand: left"
  laser-controls="hand: right"
  raycaster="objects: .interactive; showLine: true; far: 20"
  super-hands="colliderEvent: raycaster-intersection">
</a-entity>
```

#### 2. Conflitos entre Hand Tracking e Controllers

**Problema**: Quando tanto o hand tracking quanto os controladores estão ativos simultaneamente, ocorrem conflitos de interação.

**Impacto**:
- Dupla seleção de objetos
- Eventos de grab conflitantes
- Performance reduzida

#### 3. Calibração de Raycasting

**Problema**: A precisão do raycasting para seleção de objetos não está otimizada.

**Sintomas**:
- Dificuldade para selecionar botões pequenos
- Seleção de objetos não intencionais
- Distância de interação inconsistente

#### 4. Feedback Háptico Limitado

**Problema**: O feedback háptico dos controladores não está implementado.

**Impacto**:
- Experiência menos imersiva
- Dificuldade para confirmar interações
- Falta de feedback tátil

### Workarounds Temporários

1. **Recarregar a página** se os controladores não forem detectados
2. **Usar apenas hand tracking** em caso de conflitos
3. **Aumentar o tamanho dos alvos de interação** para melhor precisão
4. **Implementar feedback visual** como substituto ao háptico

### Próximos Passos para Resolução

1. **Implementar sistema de detecção robusta** de controladores
2. **Criar sistema de prioridade** entre hand tracking e controllers
3. **Otimizar configurações de raycasting** para melhor precisão
4. **Adicionar feedback háptico** para controladores compatíveis
5. **Implementar testes automatizados** para diferentes dispositivos VR

## Integração com Sistema de Drone

### Endpoints de Comunicação

&emsp;A aplicação VR se comunica com o backend do drone através de:
- **URL Base**: `http://10.140.0.11:5000`
- **Comandos**: Enviados via requisições HTTP
- **Stream de Vídeo**: Recebido via WebRTC ou HTTP streaming

### Fluxo de Dados

1. **Comando VR** → **Frontend** → **Backend Drone** → **Drone Físico**
2. **Stream Drone** → **Backend** → **Frontend VR** → **Usuário**

## Testes e Validação

### Testes Manuais

- ✅ Carregamento da interface VR
- ✅ Interação com botões via mouse (desktop)
- ⚠️ Interação com controladores VR (parcial)
- ⚠️ Hand tracking (instável)
- ✅ Redimensionamento da tela de vídeo

### Dispositivos Testados

- **Meta Quest 2**: Funcionalidade básica
- **Meta Quest 3**: Em desenvolvimento
- **Desktop Chrome**: Totalmente funcional
- **Desktop Firefox**: Funcionalidade limitada

## Conclusão

&emsp;A aplicação VR representa um avanço significativo na interface de controle de drones, oferecendo uma experiência imersiva única. Embora existam desafios técnicos com a integração de controles VR, a base está sólida e as funcionalidades core estão operacionais.

&emsp;O foco atual deve ser na resolução dos problemas de integração com controladores VR para garantir uma experiência de usuário consistente e profissional.

