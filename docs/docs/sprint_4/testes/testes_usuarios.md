---
title: Testes com os Usuários 
sidebar_position: 4
---

# Documentação dos Testes com Usuários

#### Documentação dos Testes com Usuários

**Planilha de Registro Utilizada:**
[📎 Clique aqui para acessar a planilha dos testes](https://docs.google.com/spreadsheets/d/108u0t_NM4EJ6eYzEGlCkQBQTEWgIy26T/edit?usp=sharing&ouid=103007246092712924175&rtpof=true&sd=true)

## Introdução

&emsp;Durante a Sprint 4, conduzimos uma bateria estruturada de testes com usuários para validar a experiência de uso da plataforma de detecção e classificação de fissuras. Os testes foram planejados para verificar tanto a eficácia técnica dos fluxos implementados quanto a usabilidade da interface pelos usuários finais, simulando o uso real da aplicação por diferentes perfis.

&emsp;A realização desses testes permitiu observar, com dados práticos e feedback direto, como o sistema se comporta na interação com pessoas usuárias com níveis variados de conhecimento técnico.

## Metodologia

### Participantes

&emsp;Foram selecionados sete usuários com formações e experiências acadêmicas distintas, permitindo uma análise rica e diversa sobre a interface e os fluxos do sistema:

| Nome    | Curso                    | Ano |
| ------- | ------------------------ | --- |
| Mariana | Sistemas da Informação   | 2º  |
| Nataly  | Engenharia de Software   | 2º  |
| Pablo   | Engenharia da Computação | 2º  |
| Murilo  | Engenharia da Computação | 2º  |
| Débora  | Administração Tech       | 1º  |
| Yasmin  | Engenharia de Software   | 2º  |
| Sophia  | Administração Tech       | 1º  |

### Procedimento

&emsp;Cada participante foi acompanhado individualmente e orientado a executar quatro testes principais:

* **T08:** Testar o envio automático de imagens via drone ou câmera
* **T09:** Executar os principais fluxos da interface gráfica e responder ao questionário de usabilidade SUS
* **T10:** Verificar o comportamento do sistema com diferentes níveis de permissão (admin e operador)
* **T11:** Realizar a validação, edição ou descarte de detecções feitas automaticamente

&emsp;Todas as interações foram observadas sem interferência ativa, salvo em caso de bloqueio completo. Ao fim dos testes, os dados foram registrados em planilha e as respostas ao questionário SUS foram coletadas.

## Participação e Contribuição Individual

**Mariana (Sistemas da Informação – 2º ano)**
&emsp;Demonstrou segurança na navegação pelos fluxos da aplicação. No entanto, identificou que o perfil de operador conseguiu acessar funcionalidades do painel administrativo, evidenciando uma falha no controle de permissões (T10). Também foi a única a perceber a ausência de logs após uma edição manual de detecção (T11), ponto importante para a rastreabilidade. Sugeriu melhorias no feedback visual das ações e reforçou a clareza geral da interface.

**Nataly (Engenharia de Software – 2º ano)**
&emsp;Mostrou domínio técnico ao navegar de forma eficiente e crítica. Percebeu um atraso de cerca de 14 segundos na visualização da imagem enviada automaticamente (T08), o que gerou uma leve incerteza quanto à finalização da ação. Ressaltou a boa organização dos fluxos, pontuou positivamente a coerência da interface e deu sugestões visuais pontuais. Sua avaliação SUS ficou entre as mais altas, reforçando o bom nível de usabilidade.

**Pablo (Engenharia da Computação – 2º ano)**
&emsp;Executou os testes de forma ágil e sem dificuldade. Ressaltou que o botão de “Nova Edificação” poderia ser mais evidente, pois gerou confusão com a função de criar projeto (T09). Confirmou o correto funcionamento dos acessos por perfil (T10) e elogiou o tempo de resposta do sistema. Reforçou a importância de consistência nos ícones e cores, principalmente para diferenciação de ações críticas.

**Murilo (Engenharia da Computação – 2º ano)**
&emsp;Apresentou desempenho consistente nos testes. Inicialmente demonstrou dificuldade em localizar o menu lateral e compreender sua estrutura (T09), o que levou a uma pontuação SUS ligeiramente abaixo da média. Após a ambientação, concluiu os fluxos com autonomia. Reconheceu o bom comportamento do sistema de permissões e a fluidez na validação das fissuras detectadas.

