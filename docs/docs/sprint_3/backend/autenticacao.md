---
title: Autenticação e Autorização
sidebar_position: 3
---

A presente seção documenta a implementação do mecanismo de autenticação e autorização na aplicação, baseada na utilização de JWT (JSON Web Tokens). Essa abordagem provê um modelo de autenticação stateless, no qual, após a autenticação inicial, um token assinado digitalmente é fornecido ao cliente e utilizado em requisições subsequentes, conferindo a segurança e a rastreabilidade necessárias às operações em áreas protegidas da aplicação.

## Overview

- **Login**: O processo de login é realizado mediante o fornecimento das credenciais de acesso (e-mail e senha) pelo usuário, as quais são validadas pelo backend. Em caso de sucesso, um token JWT é gerado e enviado ao cliente.
- **Proteção de Rotas**: O front-end persiste o token no `localStorage`, anexando-o a todas as requisições subsequentes que demandam autenticação.
- **Validação de Token**: O backend executa a validação do token em cada requisição direcionada a endpoints protegidos, utilizando o utilitário `JwtUtil` para verificação de integridade, assinatura e validade temporal.

## Variáveis de Ambiente

A seguir, estão listadas as variáveis de ambiente essenciais à operação segura do componente de autenticação:

```properties
# JWT
JWT_SECRET=chaveBase64de256bitsGerada
```

No ambiente do cliente (front-end), recomenda-se a configuração da seguinte variável para determinar a origem das chamadas:

```properties
REACT_APP_API_URL=http://localhost:8080
```

## Estrutura de Arquivos Relacionados

```plaintext
├── AuthController.java        # Controlador responsável pelos endpoints relacionados à autenticação
├── AuthService.java           # Serviço contendo a lógica de autenticação e autorização
├── JwtUtil.java               # Utilitário de geração e verificação de tokens JWT
├── UsuarioRepository.java     # Repositório JPA para a entidade Usuario
├── Usuario.java               # Entidade que representa o usuário e seus metadados
├── Login.jsx                  # Componente de interface de login no front-end
```

## Endpoints

<div align="center">
<sup>Quadro 1 - Endpoints do Fluxo de Autenticação</sup>

| Objetivo                         | URL                                | Método | Corpo                                                         |
| -------------------------------- | ---------------------------------- | ------ | ------------------------------------------------------------- |
| Login                            | `/auth/login`                      | POST   | ```json { "email": "usuario", "senha": "senha" } ```                     |
| Obter Usuário Autenticado        | `/auth/@me`                        | GET    | ```json Header: Authorization: Bearer {token} ```                        |

<sup>Fonte: Documentação Interna</sup>
</div>

### 1. Fluxo de Login

O procedimento de login inicia-se com o envio, por parte do cliente, de um payload JSON contendo as credenciais de acesso:

```json
{
  "email": "usuario@example.com",
  "senha": "senhaSegura123"
}
```

O backend procede com a validação dessas credenciais utilizando a camada de serviço `AuthService.java`. Quando as credenciais são validadas com sucesso, o servidor emite uma resposta contendo o JWT e um objeto simplificado com as informações públicas do usuário autenticado:

```json
{
  "token": "eyJhbGciOiJIUzI1NiJ9...",
  "usuario": {
    "id": 1,
    "email": "usuario@example.com",
    "nome": "Usuário"
  }
}
```

O token JWT é então persistido no `localStorage` pelo front-end e utilizado em todas as requisições subsequentes como forma de autenticação.

### 2. Proteção de Rotas no Front-End

No front-end, o componente `PrivateRoute` é responsável por encapsular as rotas que exigem autenticação. Ele executa a validação do token presente no `localStorage` mediante chamada ao endpoint `/auth/@me`. A lógica de verificação é ilustrada no exemplo abaixo:

```jsx
useEffect(() => {
  const checkAuth = async () => {
    try {
      await httpClient.get("/auth/@me");
      setIsAuthenticated(true);
    } catch (error) {
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  checkAuth();
}, []);
```

Se o token for considerado inválido ou expirado, o usuário é redirecionado automaticamente à interface de login para revalidação de suas credenciais.

### 3. Fluxo Geral de Autenticação

1. O cliente submete as credenciais ao endpoint de login e recebe um token JWT em caso de autenticação bem-sucedida.
2. O front-end armazena o token no `localStorage` e o inclui no cabeçalho `Authorization` das requisições subsequentes.
3. O backend valida o token recebido em cada requisição a endpoints privados, conferindo sua assinatura e validade.
4. O logout é efetivado no cliente ao remover o token do `localStorage`, invalidando a sessão do lado do cliente.
