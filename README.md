# Grupo 3 - Athena
<p align="center">
  <img src="docs/static/img/logo_athena_sf.png" alt="Logo Athena" width="300"/>
</p>

## 🎥 [Acesse o vídeo da nossa solução aqui!](https://youtu.be/i9uvU1WtSGA)

## 👨‍🎓 Integrantes:
- [Caio Santos](https://www.linkedin.com/in/caio-alcantara-santos/)
- [Carol Pascarelli](https://www.linkedin.com/in/carol-pascarelli/)
- [Cecília Galvão](https://www.linkedin.com/in/ceciliagalvaoo)
- [Heitor Candido](https://www.linkedin.com/in/heitorfariacandido/)
- [Gabriel Martins](https://www.linkedin.com/in/gabriel-martins-alves/)
- [Matheus Jorge](https://www.linkedin.com/in/matheusjorgerosa/)
- [Sophia Senne](https://www.linkedin.com/in/sophia-emanuele-de-senne-silva/)

## 👩‍🏫 Professores:

### Orientador(a)
- [Rodrigo Nicola](https://www.linkedin.com/in/rodrigo-mangoni-nicola-537027158/)

### Instrutores

- [Filipe Gonçalves](https://www.linkedin.com/in/filipe-gon%C3%A7alves-08a55015b/)
- [Geraldo Magela Severino Vasconcelos](https://www.linkedin.com/in/geraldo-magela-severino-vasconcelos-22b1b220/)
- [Guilherme Cestari](https://www.linkedin.com/in/gui-cestari/)
- [Lisane Valdo](https://www.linkedin.com/in/lisane-valdo/)
- [Murilo Zanini de Carvalho](https://www.linkedin.com/in/murilo-zanini-de-carvalho-0980415b/)
- [Rodrigo Nicola](https://www.linkedin.com/in/rodrigo-mangoni-nicola-537027158/)

## 📜 Descrição

A solução Athena tem como objetivo automatizar o processo de identificação e monitoramento de fissuras em edificações com revestimento de argamassa, oferecendo suporte direto ao trabalho técnico do IPT. Por meio de imagens capturadas por drones ou câmeras de alta resolução, o sistema aplica técnicas avançadas de processamento de imagens e inteligência artificial para detectar fissuras de forma precisa e eficiente. A partir dessas análises, são gerados automaticamente relatórios técnicos estruturados, facilitando a tomada de decisão da equipe do IPT e otimizando todo o fluxo de inspeção predial.

## 📁 Estrutura de Pastas

📦 **2025-1B-T12-EC06-G03**

```

├── .github/                  # Configurações de GitHub Actions, templates de issues, etc.
├── .idea/                    # Configurações do projeto para o IntelliJ/IDEA
├── .vscode/                  # Configurações da IDE VSCode
│
├── docs/                     # Projeto de documentação em Docusaurus
│   ├── .docusaurus/          # Build da documentação e arquivos temporários
│   ├── blog/                 # (opcional) Postagens de blog no Docusaurus
│   └── docs/                 # Onde ficam os conteúdos principais da documentação
│       ├── sprint_1/         # Documentação da Sprint 1
│       ├── sprint_2/         # Documentação da Sprint 2
│       ├── sprint_3/         # Documentação da Sprint 3
│       ├── sprint_4/         # Documentação da Sprint 4
│       ├── sprint_5/         # Documentação da Sprint 5
│       ├── como_rodar.md     # Guia completo de execução da aplicação
│       └── introducao.md     # Introdução e visão geral do projeto
│
├── src/                      # Diretório principal dos códigos-fonte da aplicação
│   ├── app/                  # Aplicativo mobile desenvolvido com Flutter
│   │   └── athenas/          # Nome do app com suporte multiplataforma
│   │       ├── android/      # Código específico para Android
│   │       ├── ios/          # Código específico para iOS
│   │       ├── lib/          # Código Dart principal do app
│   │       ├── linux/        # Configurações para Linux (Flutter desktop)
│   │       ├── macos/        # Configurações para macOS (Flutter desktop)
│   │       ├── web/          # Código gerado para build web do app
│   │       └── windows/      # Configurações para Windows (Flutter desktop)
│
│   ├── backend-modelo/       # Backend em Python para execução dos modelos de IA
│   │   ├── pycache/      # Cache interno do Python
│   │   ├── app/              # Subdiretório com virtualenv
│   │   │   └── venv/         # Ambiente virtual Python para dependências
│   │   ├── config.py         # Arquivo de configuração do servidor/modelo
│   │   ├── frontend_simulator.py # Script de teste/simulação do frontend
│   │   ├── requirements.txt  # Lista de bibliotecas necessárias (pip)
│   │   └── run.py            # Script principal para rodar o backend Python
│
│   ├── frontend/             # Código do frontend web (React + Tailwind)
│   │   ├── node_modules/     # Dependências instaladas via npm
│   │   ├── public/           # Arquivos públicos como index.html e favicon
│   │   └── src/              # Código-fonte React do frontend
│
│   ├── modelos/              # Diretório com os modelos de IA treinados
│   │   ├── modelo_segmentacao/ # Modelo para segmentar fissuras em imagens
│   │   ├── modeloB/          # Modelo classificação
│   │   └── streamlit/        # Scripts e interface de teste com Streamlit
│
│   └── vr-app/               # Aplicação de visualização de resultados com Flask
│       ├── static/           # Arquivos estáticos como CSS e JS
│       ├── templates/        # Templates HTML (Jinja2)
│       └── app.py            # Código principal do app Flask
│
├── .gitignore                # Arquivos e pastas ignorados pelo Git
├── docusaurus.config.js      # Configuração geral do site Docusaurus
├── sidebars.js               # Configuração da barra lateral do Docusaurus
├── package.json              # Dependências e scripts npm do frontend ou docs
├── package-lock.json         # Versões travadas das dependências npm
└── README.md                 # Arquivo principal de apresentação do projeto

````

## 💻 Execução do Projeto

### 🧭 Documentação — Docusaurus

Acesse a documentação online do projeto Athena:  
🔗 [Clique aqui](https://inteli-college.github.io/2025-1B-T12-EC06-G03/)

#### Rodando localmente:

```bash
git clone https://github.com/Inteli-College/2025-1B-T12-EC06-G03.git
cd docs
npm install
npm run start
````

### 🔍 Execução da Solução Completa (Java + Node.js + Python)

Para rodar backend, frontend e modelos de IA em diferentes sistemas operacionais, siga o guia:

🔗 [Como rodar o projeto](https://inteli-college.github.io/2025-1B-T12-EC06-G03/como_rodar)

## 🗃 Histórico de Lançamentos

* 0.5.0 — Finalização do projeto
* 0.4.0 — Integração e Implementação do modelo de Segmentação
* 0.3.0 — Desenvolvimento da Interface Web (Front-End e Back-End)
* 0.2.0 — Implementação do modelo de IA de Classificação
* 0.1.0 — Definição do problema e proposta da solução

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">
<a property="dct:title" rel="cc:attributionURL" href="https://github.com/Inteli-College/2025-1A-T12-EC05-G03">Athena</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.inteli.edu.br/">Inteli</a>, <a href="https://github.com/caioalcantarasantos">Caio Santos</a>, <a href="https://github.com/carolpascarelli">Carol Pascarelli</a>, <a href="https://github.com/cecilia-galvao">Cecília Galvão</a>, <a href="https://github.com/heitorcandido">Heitor Candido</a>, <a href="https://github.com/gabriemartins">Gabriel Martins</a>, <a href="https://github.com/matheusjorge">Matheus Jorge</a>, <a href="https://github.com/sophiasenne">Sophia Senne</a> is licensed under <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer">Attribution 4.0 International</a>.
</p>