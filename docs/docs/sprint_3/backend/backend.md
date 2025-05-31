---
title: Back-End
sidebar_position: 1
---

&nbsp;&nbsp;&nbsp;&nbsp;A API do back-end foi desenvolvida utilizando a linguagem [Java](https://www.java.com/pt-BR/) na versão 17.0.12 e com o framework [Spring Boot](https://spring.io/projects/spring-boot), fornecendo endpoints RESTful para gerenciamento de projetos, edifícios, empresas e imagens.

## Instruções de execução

&nbsp;&nbsp;&nbsp;&nbsp;Após clonar o repositório, dentro da pasta `src/backend` execute o arquivo `./runBackend.sh`:

``` bash
cd ./src/backend
chmod +x runBackend.sh
./runBackend.sh
```

## Variáveis de Ambiente

&nbsp;&nbsp;&nbsp;&nbsp;Crie um arquivo `.env` na raiz do back-end com as seguintes variáveis:

```properties
DB_URL=database_url
DB_USERNAME=usuario_do_banco
DB_PASSWORD=senha_do_banco
SUPABASE_PROJECT_URL=url_do_supabase
SUPABASE_BUCKET_NAME=nome_do_bucket
SUPABASE_SERVICE_ROLE_KEY=chave_do_servico
```

## Estrutura das pastas

&nbsp;&nbsp;&nbsp;&nbsp;A estrura de pastas de back-end é:

```
├── HELP.md
├── mvnw
├── mvnw.cmd
├── pom.xml
├── runBack.sh
├── run.ps1
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com.athenas.athenas
│   │   │       ├── AthenasApplication.java
│   │   │       ├── config
│   │   │       ├── controller
│   │   │       ├── dto
│   │   │       ├── model
│   │   │       ├── repository
│   │   │       ├── service
│   │   │       └── utils
│   │   └── resources
│   │       └── application.properties
│   └── test
│       └── java
│           └── com.athenas.athenas
│               └── AthenasApplicationTests.java
└── target


```

* **`config`**: Contém classes de configuração do projeto, como:
    * **`CorsConfig.java`**: define políticas de CORS para acesso à API.
    * **`GlobalExceptionHandler.java`**: captura e trata exceções globais da aplicação.
    * **`OpenApiConfig.java`**: configura a documentação da API usando Swagger/OpenAPI.

* **`controller`**: Responsável por expor os endpoints REST da API. Cada controller lida com um domínio da aplicação (ex: `ProjetoController`, `EmpresaController`, `ImageController`, etc.).

* **`dto`**: Contém os **Data Transfer Objects**, usados para trafegar dados entre o front-end e a API sem expor diretamente os modelos de banco de dados.

* **`model`**: Define as entidades do sistema, que representam as tabelas no banco de dados (ex: `Usuario`, `Projeto`, `Fissura`, etc.).

* **`repository`**: Interfaces que estendem `JpaRepository` e realizam operações de persistência no banco de dados.

* **`service`**: Contém a lógica de negócio da aplicação.

* **`utils`**: Armazena classes utilitárias. Atualmente há o `JwtUtil.java`, que lida com autenticação via JWT (JSON Web Token).

* **`resources`**: Contém arquivos de configuração, como o `application.properties`, onde são definidos parâmetros da aplicação como dados do banco de dados, porta da API, etc.

* **`test`**: Contém os testes automatizados da aplicação. Por padrão, a classe `AthenasApplicationTests.java` testa se a aplicação carrega corretamente o contexto.

## Controllers

&nbsp;&nbsp;&nbsp;&nbsp;Os controllers existentes são:

<div align="center">
<sup>Quadro 1 - Controllers</sup>

| Controller              | Responsabilidade principal                                                        |
| ----------------------- | --------------------------------------------------------------------------------- |
| `AuthController`        | Lida com autenticação de usuários (login, geração de token JWT).                  |
| `UsuarioController`     | Gerencia o CRUD de usuários do sistema.                                           |
| `EmpresaController`     | CRUD de empresas e suas associações com projetos.                                 |
| `ProjetoController`     | CRUD de projetos, incluindo vínculo com edifícios e empresas.                     |
| `EdificioController`    | Gerencia edifícios associados a projetos.                                         |
| `FissuraController`     | CRUD de fissuras detectadas em edifícios (com ou sem imagem associada).           |
| `ImageController`       | Upload, download e exclusão de imagens via Supabase.                              |
| `ViewProjetoController` | Retorna visões customizadas de projetos, agregando dados de diferentes entidades. |
| `HealthCheckController` | Endpoint simples para verificação de saúde da aplicação (status 200 OK).          |

<sup>Fonte: Material produzido pelos autores (2025)</sup>
</div>

## Endpoints

&nbsp;&nbsp;&nbsp;&nbsp;Para documentação dos endpoints foi utilizado o [Swagger](https://swagger.io/), o qual está disponível na URL abaixo quando o back-end está sendo executado.

```
http://localhost:8080/swagger-ui/index.html
```