# VR Drone App - Arquivos Separados

## Estrutura do Projeto

A aplicação VR foi reorganizada para separar o HTML, CSS e JavaScript em arquivos distintos, melhorando a manutenibilidade e organização do código.

### Arquivos Principais

#### 1. HTML
- **`templates/index_clean.html`** - Arquivo HTML principal limpo e organizado
  - Contém apenas a estrutura HTML e as referências aos arquivos externos
  - Configuração da cena A-Frame
  - Definição dos elementos VR (controladores, botões, telas)

#### 2. CSS 
- **`static/css/style.css`** - Estilos da aplicação
  - Estilos básicos do body e layout
  - Animações CSS (pulse, spin, fadeInOut)
  - Estilos para elementos interativos
  - Responsividade para diferentes tamanhos de tela
  - Estilos de debug e notificações

#### 3. JavaScript
- **`static/js/vr-components.js`** - Componentes A-Frame customizados
  - `moveable-screen` - Controla movimento e interação da tela de vídeo
  - `keyboard-controls` - Controles por teclado para desktop
  - `interactive-button` - Botões interativos para comandos do drone
  - `vr-debug` - Componente de debug para desenvolvimento
  - `simple-vr-interaction` - Simplifica interações VR

## Vantagens da Separação

### Manutenibilidade
- Código mais organizado e fácil de encontrar
- Mudanças em estilos não afetam lógica JavaScript
- Estrutura HTML mais limpa e legível

### Performance
- Arquivos CSS e JS podem ser cacheados pelo navegador
- Carregamento mais eficiente dos recursos
- Possibilidade de minificação separada

### Desenvolvimento
- Diferentes desenvolvedores podem trabalhar em arquivos diferentes
- Facilita o uso de ferramentas de desenvolvimento
- Melhor controle de versão (Git)

## Como Usar

### Desenvolvimento Local
1. Certifique-se de que os arquivos estão na estrutura correta:
   ```
   vr-app/
   ├── templates/
   │   └── index_clean.html
   ├── static/
   │   ├── css/
   │   │   └── style.css
   │   └── js/
   │       └── vr-components.js
   ```

2. Execute o servidor Flask normalmente
3. Acesse a aplicação através da rota configurada

### Modificações

#### Estilos (CSS)
- Edite `static/css/style.css` para alterar a aparência
- Adicione novas animações ou responsividade

#### Funcionalidades (JavaScript)
- Edite `static/js/vr-components.js` para alterar comportamentos
- Adicione novos componentes A-Frame
- Modifique interações VR

#### Estrutura (HTML)
- Edite `templates/index_clean.html` para alterar layout
- Adicione novos elementos VR
- Modifique configurações da cena

## Componentes JavaScript

### moveable-screen
Permite mover, rotacionar e redimensionar a tela de vídeo tanto em VR quanto com teclado.

**Controles VR:**
- Trigger/Grip: Pegar e mover
- Hand tracking: Detecção de gestos

**Controles Teclado:**
- WASD: Movimento
- Q/E: Subir/Descer
- Setas: Rotação
- +/-: Escala

### interactive-button
Botões interativos para comandos do drone.

**Funcionalidades:**
- Feedback visual (cores, animações)
- Múltiplos métodos de interação
- Notificações temporárias
- Comunicação com backend

### vr-debug
Componente de debug que registra eventos VR no console.

**Eventos monitorados:**
- Trigger down/up
- Grip down/up
- Intersecções do laser
- Hover start/end

## Próximos Passos

1. **Otimização**: Minificar arquivos CSS e JS para produção
2. **Testing**: Adicionar testes para componentes JavaScript
3. **Documentation**: Expandir documentação dos componentes
4. **Performance**: Implementar lazy loading se necessário
