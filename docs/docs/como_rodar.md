---
title: Como rodar o projeto
sidebar_label: Como rodar o projeto
sidebar_position: 8
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Como rodar o projeto

Este documento fornece instruções detalhadas para configurar e executar todas as partes do projeto em diferentes sistemas operacionais: Windows, Linux (Ubuntu) e macOS.

## Pré-requisitos gerais

Antes de começar, certifique-se de que você tem instalado:

- **Git** - Para clonar o repositório
- **Java 17** - Para o backend principal
- **Node.js e NPM** - Para o frontend
- **Python 3** - Para o backend do modelo

## Clonando o repositório

Para obter o código-fonte do projeto, clone o repositório usando Git:

```bash
# Clone o repositório
git clone https://github.com/Inteli-College/2025-1B-T12-EC06-G03

# Entre no diretório do projeto
cd 2025-1B-T12-EC06-G03
```

## Configurando e executando o Backend principal (Java)

### Instalando o Java 17

#### Windows
1. Baixe o JDK 17 do [site oficial da Oracle](https://www.oracle.com/java/technologies/downloads/#java17) ou use o OpenJDK
2. Execute o instalador e siga as instruções na tela
3. Configure as variáveis de ambiente:
   - Adicione `JAVA_HOME` apontando para o diretório de instalação do Java
   - Adicione `%JAVA_HOME%\bin` ao PATH

#### Linux (Ubuntu)
```bash
# Atualize os pacotes
sudo apt update

# Instale o OpenJDK 17
sudo apt install openjdk-17-jdk

# Verifique a instalação
java -version
```

#### macOS
```bash
# Usando Homebrew
brew install openjdk@17

# Crie um link simbólico para o sistema usar este Java
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk

# Verifique a instalação
java -version
```

### Configurando o ambiente do backend

1. Na pasta raiz do projeto (`src`), crie um arquivo `.env` baseado no `.env.example` disponível na pasta:

```bash
# Copie o arquivo de exemplo .env.example
cp src/.env.example src/.env
```

2. Entre no diretório `src/backend`:

```bash
cd src/backend
```

3. Crie outro arquivo `.env` neste diretório, seguindo o modelo do `.env.example` da pasta:

```bash
# Copie o arquivo de exemplo .env.example
cp .env.example .env
```

4. Edite ambos os arquivos `.env` para adicionar as credenciais necessárias usando seu editor de texto preferido (como VS Code, por exemplo).

### Executando o backend

#### Windows
```powershell
# Na pasta src/backend, execute o script PowerShell
.\run.ps1
```

#### Linux/macOS
```bash
# Na pasta src/backend, dê permissão de execução e rode o script bash
chmod +x runBack.sh
./runBack.sh
```

## Configurando e executando o Frontend

### Instalando Node.js

#### Windows
1. Baixe o instalador do [site oficial do Node.js](https://nodejs.org/)
2. Execute o instalador e siga as instruções na tela
3. Verifique a instalação:
```bash
node --version
npm --version
```

#### Linux (Ubuntu)
```bash
# Usando NodeSource
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verifique a instalação
node --version
npm --version
```

#### macOS
```bash
# Usando Homebrew
brew install node

# Verifique a instalação
node --version
npm --version
```

### Executando o Frontend

1. Navegue até o diretório do frontend:
```bash
cd src/frontend
```

2. Instale as dependências:
```bash
npm install
```

3. Inicie o servidor de desenvolvimento:
```bash
npm start
```

O frontend ficará disponível em `http://localhost:3000` no seu navegador.

## Configurando e executando o Backend do modelo (Python)

### Instalando Python 3

#### Windows
1. Baixe o instalador do [site oficial do Python](https://www.python.org/downloads/)
2. Execute o instalador, marque a opção "Add Python to PATH"
3. Verifique a instalação:
```bash
python --version
```

#### macOS
```bash
# Usando Homebrew
brew install python

# Verifique a instalação
python3 --version
```

### Configurando o ambiente virtual Python

1. Navegue até o diretório do backend do modelo:
```bash
cd src/backend-modelo
```

2. Instale a biblioteca venv (se ainda não estiver instalada):

#### Windows
```bash
# O venv já vem com a instalação padrão do Python em versões recentes
# Se necessário, instale com:
pip install virtualenv
```

#### Linux (Ubuntu)
```bash
sudo apt install python3-venv
```

#### macOS
```bash
pip3 install virtualenv
```

3. Crie um ambiente virtual:

#### Windows
```bash
# Usando venv
python -m venv venv

# Ou usando virtualenv
virtualenv venv
```

#### Linux (Ubuntu) / macOS
```bash
python3 -m venv venv
```

4. Ative o ambiente virtual:

#### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
venv\Scripts\activate.bat
```

#### Linux (Ubuntu) / macOS
```bash
source venv/bin/activate
```

5. Instale as dependências:

#### Windows
```bash
pip install -r requirements.txt
```

#### Linux (Ubuntu) / macOS
```bash
pip3 install -r requirements.txt
```

### Executando o Backend do Modelo

Com o ambiente virtual ativado:

#### Windows
```bash
python run.py
```

#### Linux (Ubuntu) / macOS
```bash
python3 run.py
```

## Verificando a execução

Após iniciar todos os componentes, você pode verificar se estão funcionando corretamente:

- **Backend principal**: disponível em `http://localhost:8080`
- **Frontend**: disponível em `http://localhost:3000`
- **Backend do modelo**: disponível em `http://localhost:5000`

## Solução de problemas comuns

### Problemas com Java

- Verifique se o Java 17 está instalado e configurado corretamente: `java -version`
- Certifique-se de que as variáveis de ambiente estão configuradas corretamente

### Problemas com Node.js

- Verifique se o Node.js está instalado: `node --version`
- Limpe o cache do npm: `npm cache clean --force`
- Tente reinstalar as dependências: `rm -rf node_modules && npm install`

### Problemas com Python

- Verifique se está usando o ambiente virtual correto
- Certifique-se de que todas as dependências foram instaladas corretamente
- Em caso de erros de módulo, reinstale as dependências: `pip install -r requirements.txt`

## Observações finais

- Certifique-se de que todas as credenciais nos arquivos `.env` estão configuradas corretamente
- Os scripts de inicialização podem exigir permissões adicionais em sistemas Unix (Linux/macOS)
- Para desenvolvimento, recomenda-se manter todos os componentes rodando simultaneamente
