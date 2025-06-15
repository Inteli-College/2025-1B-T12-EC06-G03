---
title: Testes Unitários dos Controllers
sidebar_position: 5
---

&nbsp;&nbsp;&nbsp;&nbsp;A realização de testes unitários para os controllers é importante principalmente quando se busca um sistema confiável, manutenível e escalável; dado que eles facilitam a implementação de mudanças com mais segurança - se os testes continuam passando, o comportamento esperado está correto. Além disso, quando um bug aparece, ter testes unitários nos controllers ajuda a entender se o problema está ali ou em outro lugar (serviço, banco, etc.). Por fim, os testes mostram o que deve acontecer quando os dados estão certos e o que deve acontecer quando estão errados (ex: 400, 404, 500 etc), facilitando o desenvolvimento de novas funcionalidades.

&nbsp;&nbsp;&nbsp;&nbsp;Nesse sentido, os testes unitários dos controllers, encontrados na pasta `src/backend/src/test/java/com/athenas/athenas/controllersTests`, foram implementados utilizando [JUnit](https://junit.org/junit5/) e [Mockito](https://site.mockito.org/). Desse modo, os testes unitários cobrem os principais cenários de cada endpoint, incluindo casos de sucesso, falha e exceção. Ademais, os mocks são configurados para simular o comportamento dos serviços e repositórios, permitindo testar o controller de forma isolada.

## Testes implementados

### 1. `EdificioControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;O arquivo `EdificioControllerTests` contém os testes para o controller `EdificioController`, testando os métodos:
    *   `createEdificio()`: Testes de sucesso, exceção e validação com `Fachada` com nome vazio.
    *   `createEdificioForProject(Long idProjeto, Edificio edificio)`: Testes de sucesso, projeto não encontrado e exceção.
    *   `createEdificioForProjectByName(String nomeProjeto, Edificio edificio)`: Testes de sucesso, projeto não encontrado e exceção.
    *   `getAllEdificios()`: Testes de sucesso, sem conteúdo e exceção.
    *   `getEdificioById(Long id)`: Testes de sucesso, não encontrado e exceção.
    *   `getAllEdificiosByProjectName(String nomeProjeto)`: Testes de sucesso, projeto não encontrado, sem conteúdo e exceção.
    *   `getAllEdificiosByProject(Long idProjeto)`: Testes de sucesso, projeto não encontrado e exceção.
    *   `updateEdificio(Long id, Edificio edificio)`: Testes de sucesso, não encontrado e exceção.
    *   `deleteEdificio(Long id)`: Testes de sucesso, não encontrado e exceção.
    *   `deleteAllEdificiosByProject(Long idProjeto)`: Testes de sucesso, projeto não encontrado e exceção.

### 2. `EmpresaControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `EmpresaController`. Testa os métodos:
    *   `createEmpresa(EmpresaDTO empresaDTO)`: Testes de sucesso e exceção.
    *   `getAllEmpresas()`: Testes de sucesso, lista vazia e exceção.
    *   `getEmpresaById(Integer id)`: Testes de sucesso, não encontrado e exceção.
    *   `getEmpresaByNome(String nome)`: Testes de sucesso, não encontrado e exceção.
    *   `getEmpresaByCNPJ(String cnpj)`: Testes de sucesso, não encontrado e exceção.
    *   `updateEmpresa(Long id, EmpresaDTO empresaDTO)`: Testes de sucesso, não encontrado e exceção.
    *   `deleteEmpresa(Long id)`: Testes de sucesso, exceção e empresa não encontrada.

### 3. `FissuraControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `FissuraController`. Testa os métodos:
    *   `createFissura(Fissura fissura)`: Testes de sucesso, imagem nula, ID da imagem nulo, imagem não encontrada, sem data de detecção e exceção.
    *   `aprovarFissura(Long id, Map<String, Object> request)`: Testes de sucesso, não encontrado e exceção.
    *   `getPorcentagemPorTipo(Integer projetoId)`: Teste de sucesso.
    *   `getFissurasByProjeto(Integer projetoId)`: Testes de sucesso e projeto não encontrado.
    *   `getFissurasDetalhadasByProjeto(Integer projetoId)`: Testes de sucesso, projeto não encontrado e com data de detecção nula.
    *   `getFissurasDetalhesByImagem(Long imagemId)`: Testes de sucesso, imagem não encontrada e lista de fissuras vazia.

### 4. `ImageControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `ImageController`. Testa os métodos:
    *   `uploadFiles(Long projetoId, String lado, Long fachadaId, List<MultipartFile> files)`: Testes de sucesso, projeto não encontrado, exceção e lista de arquivos vazia, e upload de um único arquivo.
    *   `getImagesByProjectId(Long projetoId)`: Testes de sucesso, projeto não encontrado e lista vazia.
    *   `updateImageProcessada(Long id, ProcessadaRequest processadaRequest)`: Testes de sucesso, imagem não encontrada, com `processada` falso e com valores nulos.
    *   `deleteImage(Long id)`: Testes de sucesso, imagem não encontrada e exceção.
    *   `ProcessadaRequest`: Testes para os getters e setters e valores nulos.

### 5. `ProjetoControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `ProjetoController`. Testa os métodos:
    *   `listAllProjects(String nome)`: Testes com e sem nome, com nome vazio e com nome sem resultados.
    *   `createProject(ProjetoDTO projetoDTO)`: Testes de sucesso e com valores nulos.
    *   `getProjectById(Long id)`: Testes de sucesso e projeto não encontrado.
    *   `updateProject(Long id, ProjetoDTO projetoDTO)`: Testes de sucesso, projeto não encontrado e com valores nulos.
    *   Testes adicionais para cenários de exceção nos métodos `createProject`, `getProjectById` e `updateProject`.

### 6. `UsuarioControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `UsuarioController`. Testa os métodos:
    *   `getAllUsuarios()`: Testes de sucesso, lista vazia e exceção.
    *   `getUsuarioById(Long id)`: Testes de sucesso e não encontrado.
    *   `getUsuarioByEmail(String email)`: Testes de sucesso e não encontrado.
    *   `createUsuario(Usuario usuario)`: Testes de sucesso e com valores nulos.
    *   `updateUsuario(Long id, Usuario usuario)`: Testes de sucesso e não encontrado.
    *   `deleteUsuario(Long id)`: Testes de sucesso e não encontrado.
    *   `updateUltimoAcesso(Long id)`: Testes de sucesso e não encontrado.
    *   Testes adicionais para cenários de exceção nos métodos `getAllUsuarios`, `getUsuarioById`, `getUsuarioByEmail`, `createUsuario`, `updateUsuario`, `deleteUsuario` e `updateUltimoAcesso`.

### 7. `ViewProjetoControllerTests`
&nbsp;&nbsp;&nbsp;&nbsp;Testes para o controller `ViewProjetoController`.
    *   `viewProjeto(ViewProjetoRequestDTO viewProjetoRequestDTO)`: Testes de sucesso, exceção, serviço retornando nulo e ID do projeto inválido.
    *   `updateViewProjeto(UpdateViewProjetoRequest updateViewProjetoRequest)`: Testes de sucesso, serviço retornando nulo, projeto não encontrado, requisição nula, exceção e atualização parcial.

## Resultados obtidos
&nbsp;&nbsp;&nbsp;&nbsp;Nesse cenário, os testes unitários para os controllers do projeto foram executados com 100% de sucesso, sendo que o relatório de execução totalizou 134 testes rodados, com 0 falhas, 0 erros e 0 testes ignorados.

&nbsp;&nbsp;&nbsp;&nbsp;Outrossim, para avaliar a qualidade da cobertura de código, foi utilizada a ferramenta [JaCoCo](https://www.eclemma.org/jacoco/); assim, os resultados demonstram uma excelente cobertura para a maioria dos controllers, conforme detalhado no Quadro 1 abaixo.

<div align="center">
<sup>Quadro 1 - Cobertura dos Testes</sup>

| Controller | Instruções Cobertas | Linhas Cobertas | Métodos Cobertos |
|------------|--------------------|-----------------|-----------------| 
| FissuraController | 404/404 (100%) | 82/82 (100%) | 12/12 (100%) |
| EdificioController | 435/447 (97%) | 98/101 (97%) | 11/11 (100%) |
| ImageController | 202/202 (100%) | 40/40 (100%) | 6/6 (100%) |
| UsuarioController | 92/92 (100%) | 23/23 (100%) | 10/10 (100%) |
| ProjetoController | 92/92 (100%) | 21/21 (100%) | 6/6 (100%) |
| EmpresaController | 42/42 (100%) | 11/11 (100%) | 8/8 (100%) |
| ViewProjetoController | 28/28 (100%) | 8/8 (100%) | 3/3 (100%) |

<sup>Fonte: Material Produzido pelos autores (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;Portanto, a análise dos dados revela que seis dos sete controllers alcançaram 100% de cobertura em instruções, linhas e métodos. O `EdificioController` obteve uma cobertura de 97% em instruções e linhas, com 100% dos métodos cobertos. Logo, essa alta porcentagem de cobertura demonstra a eficácia dos testes unitários em validar o comportamento dos controllers nos diversos cenários.



