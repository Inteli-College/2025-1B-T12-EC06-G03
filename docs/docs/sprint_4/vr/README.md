# Documentação VR App

Esta seção contém toda a documentação relacionada à aplicação de realidade virtual (VR) para controle de drone desenvolvida no Sprint 4.

## Estrutura da Documentação

### [doc_vr.md](doc_vr.md)
**Documentação Principal**
- Visão geral da aplicação VR
- Arquitetura e tecnologias utilizadas
- Funcionalidades implementadas
- **Seção detalhada sobre problemas de integração com controles VR**
- Status atual e próximos passos

### [setup-desenvolvimento.md](setup-desenvolvimento.md)
**Guia de Setup**
- Configuração do ambiente de desenvolvimento
- Instalação de dependências
- Configuração HTTPS para WebXR
- Setup para diferentes dispositivos VR
- Troubleshooting comum

### [componentes-aframe.md](componentes-aframe.md)
**Componentes Customizados**
- Documentação técnica dos componentes A-Frame
- `moveable-screen`: Tela interativa de vídeo
- `interactive-button`: Botões de comando
- `vr-debug`: Sistema de debug
- Exemplos de uso e boas práticas

### [problemas-integracao.md](problemas-integracao.md)
**Análise de Problemas VR**
- **Detecção inconsistente de controladores**
- **Conflitos entre hand tracking e controllers**
- **Problemas de precisão no raycasting**
- **Ausência de feedback háptico**
- Soluções propostas e workarounds

### [api-integracao.md](api-integracao.md)
**Integração com Drone**
- Documentação da API REST
- Endpoints disponíveis (takeoff, land, battery, flip)
- Configuração do stream de vídeo
- Tratamento de erros e retry logic
- Monitoramento de conexão

## Status Atual

### Funcionalidades Implementadas
- Interface VR básica funcionando
- Controles por teclado (desktop)
- Botões interativos para comandos do drone
- Stream de vídeo (placeholder)
- Sistema de debug básico

### Problemas Conhecidos
- **Detecção de controladores VR instável**
- **Conflitos entre sistemas de input**
- **Raycasting precisa de otimização**
- **Feedback háptico não implementado**

### Em Desenvolvimento
- Sistema robusto de detecção de controladores
- Resolução de conflitos hand tracking vs controllers
- Otimização de performance
- Testes em dispositivos reais

## Como Começar

1. **Leia primeiro**: [doc_vr.md](doc_vr.md) para entender a aplicação
2. **Configure o ambiente**: [setup-desenvolvimento.md](setup-desenvolvimento.md)
3. **Entenda os problemas**: [problemas-integracao.md](problemas-integracao.md)
4. **Explore os componentes**: [componentes-aframe.md](componentes-aframe.md)
5. **Configure a API**: [api-integracao.md](api-integracao.md)

## Desenvolvimento

### Estrutura do Projeto VR
```
src/vr-app/
├── app.py                 # Servidor Flask
├── templates/
│   └── index.html        # Interface VR principal  
└── static/              # Recursos estáticos
```

### Tecnologias Principais
- **A-Frame 1.6.0**: Framework WebXR
- **WebXR**: APIs de realidade virtual/aumentada
- **Super Hands**: Sistema de interação 3D
- **Flask**: Backend Python

### Dispositivos Testados
- Desktop (Chrome/Firefox)
- Meta Quest 2 (funcionalidade parcial)
- Meta Quest 3 (em teste)

## Suporte

Para questões específicas sobre a implementação VR:

1. **Consulte os problemas conhecidos** em [problemas-integracao.md](problemas-integracao.md)
2. **Verifique o setup** em [setup-desenvolvimento.md](setup-desenvolvimento.md)
3. **Analise os logs** usando o sistema de debug integrado

## Demo

Para testar a aplicação:

```bash
cd src/vr-app
python app.py
# Acesse: https://localhost:5000
```

**Nota**: HTTPS é obrigatório para funcionalidades WebXR.
