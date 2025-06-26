# Documentação sobre as Modificações e Integração no Frontend

Na presente Sprint, diversas modificações substanciais foram implementadas no componente de **Cadastro** da aplicação, com ênfase na **integração plena** entre o frontend e o backend, de acordo com as especificações requeridas no desenvolvimento da sprint. A seguir, são detalhadas as principais alterações efetuadas, incluindo a implementação de funcionalidades que asseguram a comunicação fluida entre a interface de usuário e o serviço de backend, além dos testes realizados para garantir a qualidade do código implementado.

## 1. Integração com Backend (Cadastro)

A principal alteração nesta sprint reside na **integração do fluxo de cadastro** com o backend. Anteriormente, o componente de cadastro apenas coletava os dados do usuário sem realizar a devida persistência no banco de dados através do serviço de backend. Com a modificação, a aplicação agora utiliza o cliente HTTP (`httpClient`) para fazer uma requisição `POST` ao endpoint `/auth/register` da API, o qual aceita os dados do usuário e os registra no sistema.

A requisição `POST` inclui os seguintes parâmetros no corpo da mensagem (payload):
- **nome**: Nome completo do usuário.
- **email**: Endereço de e-mail do usuário.
- **senha**: Senha do usuário, que é codificada antes de ser enviada para o servidor.
- **cargo**: Cargo do usuário, com o objetivo de categorizar os usuários no sistema.

O backend, por sua vez, processa esses dados de acordo com a lógica de validação de registros existentes, codificando a senha utilizando o algoritmo `BCrypt`, conforme esperado pela arquitetura de segurança estabelecida. Caso o cadastro seja bem-sucedido, o frontend é redirecionado automaticamente para a página de login. Caso contrário, um erro é exibido na interface de usuário.

## 2. Gerenciamento de Estado e Feedback Visual

O gerenciamento de estado foi aprimorado para proporcionar uma experiência de usuário mais fluida e informativa. Foram introduzidos dois estados adicionais:
- **loading**: Um estado booleano que gerencia a exibição de um indicador de carregamento enquanto o processo de envio dos dados ao backend está em andamento. Isso proporciona ao usuário uma percepção clara de que o sistema está processando sua solicitação.
- **error**: Um estado de erro que captura e exibe mensagens detalhadas quando uma falha ocorre no backend ou na validação de dados, como erros de validação do servidor ou falhas na comunicação HTTP.

Durante o processo de submissão, a interface de usuário desativa os campos de entrada e o botão de envio, evitando interações desnecessárias enquanto o sistema processa a requisição. O feedback de erro, caso necessário, é visualmente destacado com uma tipografia em vermelho, proporcionando clareza ao usuário sobre os problemas enfrentados durante a operação.

## 3. Testes e Garantias de Qualidade

Com o foco em garantir a integridade do sistema como um todo, esta sprint também foi marcada por uma **ampliação substancial na cobertura de testes** no frontend. Testes unitários e de integração foram executados rigorosamente para verificar a funcionalidade dos componentes de entrada, validação de dados e manipulação de estados, como a exibição de mensagens de erro e feedback de carregamento. Esses testes visam garantir que todos os fluxos de interação do usuário sejam tratados de maneira robusta e sem falhas, tanto em cenários de sucesso quanto de erro.

Particularmente, a integração com o backend foi exaustivamente testada para assegurar que a comunicação entre o frontend e o serviço de autenticação/registro funcione conforme esperado, sem perdas de dados ou inconsistências na comunicação.

## 4. Redirecionamento após Cadastro Bem-Sucedido

Uma característica crítica foi a implementação de **redirecionamento automático** do usuário para a página de login após o registro bem-sucedido. Isso foi feito através da utilização do método `window.location.href`, que altera a localização do navegador para a página de login assim que o backend confirma que o registro foi realizado sem erros. Essa abordagem garante uma navegação fluida e sem a necessidade de interações adicionais por parte do usuário.

## 5. Conclusão e Integração Completa

Com estas modificações, o fluxo de cadastro agora está **inteiramente integrado** com o backend, funcionando de maneira coesa e alinhada com as expectativas da arquitetura da aplicação. O frontend e o backend estão interconectados, permitindo que os dados do usuário sejam capturados corretamente e transmitidos de forma segura para o servidor, garantindo tanto a persistência quanto a segurança das informações.

Além disso, foram realizados mais testes no frontend, assegurando que todas as interações e fluxos do sistema estejam operando de acordo com o esperado. **Agora, o sistema está devidamente integrado da maneira que deveria estar, com todos os componentes interagindo de forma eficiente e sem falhas**. A experiência do usuário foi aprimorada com feedbacks claros e visuais durante o processo de registro, proporcionando maior confiabilidade e usabilidade da aplicação.

## Resumo das Alterações
- **Integração do frontend com o backend** para registro de usuários.
- **Implementação de estados de carregamento e erro** para melhorar a experiência de usuário.
- **Ampliação da cobertura de testes**, com ênfase na validação da integração entre frontend e backend.
- **Redirecionamento pós-cadastro** para a página de login após sucesso no cadastro.
- **Garantia de funcionamento coeso e alinhado** entre os sistemas de frontend e backend, proporcionando uma operação fluida e sem falhas.
