---
title: Integração com o Back-End
sidebar_position: 2
---

&nbsp;&nbsp;&nbsp;&nbsp;A integração entre o front-end (desenvolvido em React) e o back-end foi realizada por meio de chamadas HTTP às rotas da API. Essas chamadas foram implementadas principalmente com a função `fetch`, sendo executadas dentro do hook `useEffect` para garantir a execução no momento apropriado do ciclo de vida do componente. Nesse cenário, as telas que foram integradas são:


### 1. Projetos

&nbsp;&nbsp;&nbsp;&nbsp;A tela de Projetos permite cadastro, visualização e busca de projetos cadastrados no sistema. As funcionalidades disponíveis são:

* Listar Projetos: Carrega todos os projetos cadastrados no sistema, mostrando nome, empresa vinculada, descrição e status.
    * Requisição: `GET /api/projetos`
    * Exemplo de corpo da resposta:
    ```JSON
    [{
        "id": 1,
        "nome": "USP",
        "empresa": {
            "id": 1,
            "nome": "Construtora Alpha",
            "cnpj": "12345678000190",
            "endereco": "Av. Paulista, 1000, São Paulo, SP",
            "telefone": "(11) 3456-7890",
            "email": "contato@alpha.com.br"
        },
        "descricao": "O projeto busca achar rachaduras nos prédios da USP.",
        "dataCriacao": "2025-05-20T17:48:40.653942",
        "dataAtualizacao": "2025-05-20T17:48:40.653778",
        "status": "EM_ANDAMENTO"
    }]
    ```

* Listar Empresas: Carrega todas as empresas disponíveis para vincular a novos projetos.
    * Requisição: `GET /api/empresa/getEmpresas`
    * Exemplo de corpo da resposta:
    ```JSON
    [{
        "id": 1,
        "nome": "Construtora Alpha",
        "cnpj": "12345678000190",
        "endereco": "Av. Paulista, 1000, São Paulo, SP",
        "telefone": "(11) 3456-7890",
        "email": "contato@alpha.com.br"
    }]
    ```

* Cadastrar Novo Projeto: Formulário com os seguintes campos: Nome, Empresa (seleção), Descrição e Status.
    * Requisição: `POST /api/projetos`
    * Exemplo de corpo da requisição:
    ```JSON
    {
        "nome": "Novo Projeto",
        "empresa": 1,
        "descricao": "Descrição do novo projeto",
        "status": "EM_ANDAMENTO"
    }
    ```
&nbsp;&nbsp;&nbsp;&nbsp;Portanto, as requisições feitas por essa página podem ser visualizadas no Quadro 1 abaixo.

<div align="center">
<sup>Quadro 1 - Requisições Projetos</sup>

| Objetivo                         | URL                                 | Método | Corpo                        |
| ---------------------------- | ----------------------------------- | ------ | ---------------------------- |
| **Buscar todos projetos** | `/api/projetos` | GET    | N/a                            |
| **Buscar todas empresas** | `/api/empresa/getEmpresas` | GET    | N/a                            |
| **Criar projeto**               | `/api/projetos` | POST   | Dados coletados nos campos do formulário transformados em JSON (nome, empresa, descrição, status). |

<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

### 2. Clientes

&nbsp;&nbsp;&nbsp;&nbsp;A tela de Clientes permite cadastro, edição, visualização, busca e exclusão de empresas/clientes cadastrados no sistema. As funcionalidades disponíveis são:

* Listar Clientes: Carrega todos os clientes/empresas cadastradas no sistema, mostrando nome, CNPJ, endereço, telefone e e-mail.
    * Requisição: `GET /api/empresa/getEmpresas`
    * Exemplo de corpo da resposta:
    ```JSON
    [{
        "id": 1,
        "nome": "Construtora Alpha",
        "cnpj": "12345678000190",
        "endereco": "Av. Paulista, 1000, São Paulo, SP",
        "telefone": "(11) 3456-7890",
        "email": "contato@alpha.com.br"
    }]
    ```

* Cadastrar Novo Cliente: Formulário com os seguintes campos: Nome, CNPJ, Endereço, Telefone e E-mail.
    * Requisição: `POST /api/empresa/create`
    * Exemplo de corpo da requisição:
    ```JSON
    {
        "nome": "Construtora Beta",
        "cnpj": "98765432000198",
        "endereco": "Av. Faria Lima, 1500, São Paulo, SP",
        "telefone": "(11) 9876-5432",
        "email": "contato@beta.com.br"
    }
    ```

* Editar Cliente: Ao clicar no ícone de lápis, carrega os dados no formulário permitindo a edição das informações.
    * Requisição: `PUT /api/empresa/update/{id}`, sendo `id` o ID do cliente no banco de dados.
    * Exemplo de corpo da requisição:
    ```JSON
    {
        "nome": "Construtora Alpha Atualizada",
        "cnpj": "12345678000190",
        "endereco": "Av. Paulista, 2000, São Paulo, SP",
        "telefone": "(11) 3456-7890",
        "email": "contato@alpha.com.br"
    }
    ```

* Excluir Cliente: Ao clicar no ícone de lixeira, abre-se um modal para confirmação da exclusão.
    * Requisição: `DELETE /api/empresa/delete/{id}`, sendo `id` o ID do cliente no banco de dados.

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, as requisições feitas por essa página podem ser visualizadas no Quadro 2 abaixo.

<div align="center">
<sup>Quadro 2 - Requisições Clientes</sup>

| Objetivo                         | URL                                 | Método | Corpo                        |
| ---------------------------- | ----------------------------------- | ------ | ---------------------------- |
| **Buscar todos clientes** | `/api/empresa/getEmpresas` | GET    | N/a                            |
| **Criar cliente**               | `/api/empresa/create` | POST   | Dados coletados nos campos do formulário (nome, CNPJ, endereço, telefone, e-mail) transformados em JSON. |
| **Atualizar cliente**           | `/api/empresa/update/{id}`                | PUT    | Campos do formulário transformados em JSON. |
| **Excluir cliente**             | `/api/empresa/delete/{id}`                | DELETE | N/a                            |

<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

### 3. Edifícios

&nbsp;&nbsp;&nbsp;&nbsp;A tela de Edifícios permite cadastro, edição, visualização, busca e exclusão de edifícios vinculados a um projeto específico, identificado pelo parâmetro `?projeto=NOME_DO_PROJETO` na URL (exemplo: `http://localhost:3000/edificios?projeto=MeuProjeto`). Desse modo, a fim de permitir essas ações, as funcionalidades disponiveis são:

* Listar Edifícios: Carrega todos edifícios vinculados ao projeto informado na URL, mostrando nome, localização, tipo, pavimentos e fachadas.
    * Requisição: `GET /api/edificio/projeto-nome/{nome}`, sendo `nome`, o nome do edifício recebido como parâmetro na URL.
    * Exemplo de corpo da resposta:
    ```JSON
    [{
        "id":1,
        "projeto":{
            "id":1,
            "nome":"USP",
            "empresa":{
                "id":1,
                "nome":"Construtora Alpha",
                "cnpj":"12345678000190",
                "endereco":"Av. Paulista, 1000, São Paulo, SP",
                "telefone":"(11) 3456-7890",
                "email":"contato@alpha.com.br"
            },
            "descricao":"O projeto busca achar rachaduras nos prédios da USP.",
            "dataCriacao":"2025-05-20T17:48:40.653942",
            "dataAtualizacao":"2025-05-20T17:48:40.653778",
            "status":"EM_ANDAMENTO"
        },
        "nome":"Bloco A",
        "localizacao":"Rua das Flores, 100",
        "tipo":"Residencial",
        "pavimentos":10
    }]
    ```

* Cadastrar Novo Edifício Formulário com os seguintes campos: Nome, Localização, Tipo, Pavimentos (número) e Fachadas (área e descrição).
    * Requisição: `POST /api/edificio/projeto-nome/{projeto}`, dado que `projeto` é o nome do projeto ao qual o edifício está vinculado.
    * Exemplo de corpo da requisição:

    ```JSON
    {
        "id":18,
        "projeto":{
            "id":1,
            "nome":"USP",
            "empresa":{
                "id":1,
                "nome":"Construtora Alpha",
                "cnpj":"12345678000190",
                "endereco":"Av. Paulista, 1000, São Paulo, SP",
                "telefone":"(11) 3456-7890",
                "email":"contato@alpha.com.br"
            },
            "descricao":"O projeto busca achar rachaduras nos prédios da USP.",
            "dataCriacao":"2025-05-20T17:48:40.653942",
            "dataAtualizacao":"2025-05-20T17:48:40.653778",
            "status":"EM_ANDAMENTO"
        },
        "nome":"Prédio 25",
        "localizacao":"Av. 9 de julho, 5030, Vila Emma",
        "tipo":"Comercial",
        "pavimentos":20,
        "fachadas":[
            {"area":500,"descricao":"Norte"},
            {"area":500,"descricao":"Sul"},
            {"area":700,"descricao":"Leste"},
            {"area":700,"descricao":"Oeste"}
        ]
    }

    ```

* Editar Edifício: Ao clicar no ícone de lápis, carrega os dados no formulário permitindo a edição das informações.
    * Requisição: `PUT /api/edificio/{id}`, sendo `id`, a id do edifício no banco de dados.
    * Exemplo de corpo da requisição:

    ```JSON
    {
        "nome":"Prédio 25",
        "localizacao":"Av. 9 de julho, 5000, Vila Addyana",
        "tipo":"Comercial",
        "pavimentos":20,
        "fachadas":[
            {"area":500,"descricao":"Norte"},
            {"area":500,"descricao":"Sul"},
            {"area":700,"descricao":"Leste"},
            {"area":700,"descricao":"Oeste"}
        ],
        "projeto":{
            "id":1,
            "nome":"USP",
            "empresa":{
                "id":1,
                "nome":"Construtora Alpha",
                "cnpj":"12345678000190",
                "endereco":"Av. Paulista, 1000, São Paulo, SP",
                "telefone":"(11) 3456-7890",
                "email":"contato@alpha.com.br"
            },
            "descricao":"O projeto busca achar rachaduras nos prédios da USP.",
            "dataCriacao":"2025-05-20T17:48:40.653942",
            "dataAtualizacao":"2025-05-20T17:48:40.653778",
            "status":"EM_ANDAMENTO"
        }
    }

    ```

* Excluir Edifício: Ao clicar no ícone de lixeira, abre-se um modal para confirmação da exclusão e, após a confirmação, a requisição é feita passando o id do edifício como parâmetro na rota.
    * Requisição: `DELETE /api/edificio/{id}`

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, as requisições feitas por essa página podem ser visualizadas no Quadro 3 abaixo.

<div align="center">
<sup>Quadro 3 - Requisições Edifícios</sup>

| Objetivo                         | URL                                 | Método | Corpo                        |
| ---------------------------- | ----------------------------------- | ------ | ---------------------------- |
| **Buscar edifícios por projeto** | `/api/edificio/projeto-nome/{nome}` | GET    | N/a                            |
| **Criar edifício**               | `/api/edificio/projeto-nome/{nome}` | POST   | Dados coletados nos campos do formulario transformados em JSON.          |
| **Atualizar edifício**           | `/api/edificio/{id}`                | PUT    | Campos do formulaŕio mais o projeto vinculado transformados em JSON. |
| **Excluir edifício**             | `/api/edificio/{id}`                | DELETE | N/a                            |

<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

#### 4. Relatório

&nbsp;&nbsp;&nbsp;&nbsp;A tela de Relatório permite visualizar, editar e exportar relatórios de um projeto específico, identificado pelo parâmetro `?projeto=NOME_DO_PROJETO` na URL (exemplo: `http://localhost:3000/visualizar?projeto=MeuProjeto`). As funcionalidades disponíveis são:

* Buscar Projeto: Carrega os dados do projeto pelo nome informado na URL.
    * Requisição: `GET /api/projetos?nome={nome}`
    * Exemplo de corpo da resposta:
    ```JSON
    [{
        "id": 1,
        "nome": "USP",
        "empresa": {
            "id": 1,
            "nome": "Construtora Alpha",
            "cnpj": "12345678000190",
            "endereco": "Av. Paulista, 1000, São Paulo, SP",
            "telefone": "(11) 3456-7890",
            "email": "contato@alpha.com.br"
        },
        "descricao": "O projeto busca achar rachaduras nos prédios da USP.",
        "dataCriacao": "2025-05-20T17:48:40.653942",
        "dataAtualizacao": "2025-05-20T17:48:40.653778",
        "status": "EM_ANDAMENTO"
    }]
    ```

* Visualizar Detalhes do Projeto: Exibe informações completas do projeto.
    * Requisição: `POST /api/projeto/ViewProjeto` (envia o ID do projeto)
    * Exemplo de corpo da requisição:
    ```JSON
    {
        "idProjeto": 1
    }
    ```
    * Exemplo de corpo da resposta:
    ```JSON
    {
        "projeto":"USP",
        "responsaveis":["Administrador"],
        "empresa":"Construtora Alpha",
        "edificios":[
            {
                "nome":"Bloco A",
                "localizacao":"Rua das Flores, 100",
                "tipo":"Residencial","pavimentos":10
            },
            {
                "nome":"Bloco B",
                "localizacao":"Rua das Flores, 100",
                "tipo":"Residencial","pavimentos":8
            },
            {
                "nome":"Prédio 25",
                "localizacao":"Av. 9 de julho, 5000, Vila Addyana",
                "tipo":"Comercial",
                "pavimentos":20
            }],
        "descricao":"O projeto busca achar rachaduras nos prédios da USP.",
        "logs_alteracoes":["15/05/2025 19:00:20 - Atualizou status da fachada","19/05/2025 19:00:20 - Ajustou coordenadas de fissura","21/05/2025 23:21:28 - Responsável Administrador removido dos responsáveis pelo projeto.","21/05/2025 23:21:29 - Responsável Engenheiro removido dos responsáveis pelo projeto."]
    }
    ```

* Estatísticas de Fissuras: Mostra porcentagem de tipos de fissuras.
    * Requisição: `GET /api/fissura/porcentagem/{idProjeto}`
    * Exemplo de corpo da resposta:
    ```JSON
    {
        "porcentagemPorTipo":{
            "Térmica":100
        }
    }
    ```

* Detalhes das Fissuras: Lista todas as fissuras do projeto com imagens.
    * Requisição: `GET /api/fissura/detalhes/projeto/{idProjeto}`
    * Exemplo de corpo da resposta:
    ```JSON
    [
        {
            "id":2,
            "tipo":"Térmica",
            "coordenadas":"{\"x\":300,\"y\":150,\"w\":80,\"h\":20}",
            "gravidade":"Alta",
            "dataDeteccao":"2025-05-15T19:00:20.886720",
            "confianca":0.92,
            "nomeImagem":"0001/east/FT81.png",
            "porcentagemPorTipo":null,
            "processada":null
        },
        {
            "id":1,
            "tipo":"Térmica",
            "coordenadas":"{\"x\":100,\"y\":200,\"w\":50,\"h\":10}",
            "gravidade":"Baixa",
            "dataDeteccao":"2025-05-14T19:00:20.886720",
            "confianca":0.85,
            "nomeImagem":"0001/east/FT80.png",
            "porcentagemPorTipo":null,
            "processada":null
        }
    ]
    ```

* Editar Projeto: Permite alterar responsáveis, edifícios e descrição.
    * Requisição: `PUT /api/projeto/UpdateViewProjeto`
    * Exemplo de corpo da requisição:
    ```JSON
    {
        "projeto":"USP",
        "responsaveis":["Administrador"],
        "empresa":"Construtora Alpha",
        "edificios":[
            {
                "nome":"Bloco A",
                "localizacao":"Rua das Flores, 100",
                "tipo":"Residencial",
                "pavimentos":10
            },
            {
                "nome":"Bloco B",
                "localizacao":"Rua das Flores, 100",
                "tipo":"Residencial",
                "pavimentos":8
            }
        ],
        "descricao":"O projeto busca achar rachaduras nos prédios da USP.",
        "logs_alteracoes":["15/05/2025 19:00:20 - Atualizou status da fachada","19/05/2025 19:00:20 - Ajustou coordenadas de fissura","21/05/2025 23:21:28 - Responsável Administradorremovido dos responsáveis pelo projeto.","21/05/2025 23:21:29 - Responsável Engenheiroremovido dos responsáveis pelo projeto.","21/05/2025 23:23:50 - Responsável Engenheiroremovido dos responsáveis pelo projeto.","21/05/2025 23:24:57 - Responsável 'Administrador' removido dos responsáveis pelo projeto."]
    }
    ```

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, as requisições feitas por essa página podem ser visualizadas no Quadro 4 abaixo.

<div align="center">
<sup>Quadro 4 - Requisições Visualização de Projeto</sup>

| Objetivo                         | URL                                 | Método | Corpo                        |
| ---------------------------- | ----------------------------------- | ------ | ---------------------------- |
| **Buscar projeto por nome** | `/api/projetos?nome={nome}` | GET    | N/a                            |
| **Visualizar detalhes** | `/api/projeto/ViewProjeto` | POST   | ID do projeto em JSON          |
| **Estatísticas de fissuras** | `/api/fissura/porcentagem/{idProjeto}` | GET    | N/a                            |
| **Detalhes das fissuras** | `/api/fissura/detalhes/projeto/{idProjeto}` | GET    | N/a                            |
| **Atualizar projeto** | `/api/projeto/UpdateViewProjeto` | PUT    | ID do projeto e dados atualizados em JSON |

<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>



