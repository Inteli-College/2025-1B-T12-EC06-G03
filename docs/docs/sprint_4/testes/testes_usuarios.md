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

**Débora (Administração Tech – 1º ano)**
&emsp;Mesmo com menor familiaridade com sistemas técnicos, conseguiu executar todos os fluxos com sucesso. Reportou insegurança no momento do envio automático da imagem (T08), devido à ausência de feedback imediato indicando que a ação havia sido concluída. Considerou os ícones autoexplicativos e o processo de revisão das fissuras muito direto. A interface foi bem avaliada, especialmente pelo layout limpo e funcional.

**Yasmin (Engenharia de Software – 2º ano)**
Trouxe contribuições importantes sobre a usabilidade geral. Apontou que, embora o envio automático tenha funcionado corretamente, a ausência de confirmação visual (ex: mensagem de sucesso ou animação) pode gerar confusão (T08). Sugeriu a adição de uma etapa de confirmação no botão de exclusão de detecção. Teve boa experiência nos testes de permissão e considerou a navegação rápida e intuitiva.

**Sophia (Administração Tech – 1º ano)**
&emsp;Conseguiu executar todos os fluxos sem ajuda, mas teve dificuldade para localizar o botão de criação de edificação, que estava fora do campo de visão principal (T09). Essa dificuldade impactou sua avaliação no questionário SUS, com nota ligeiramente abaixo da média. No restante dos testes, executou ações de forma tranquila e destacou a clareza das opções no processo de validação manual.

## Questionário SUS (System Usability Scale)

&emsp;Ao final dos testes, foi aplicado o questionário padrão de usabilidade SUS com as seguintes 10 perguntas, respondidas em escala de 1 (discordo totalmente) a 5 (concordo totalmente):

1. Eu acho que gostaria de usar esse sistema com frequência.
2. Eu achei o sistema desnecessariamente complexo.
3. Eu achei o sistema fácil de usar.
4. Eu acho que precisaria de ajuda técnica para usar esse sistema.
5. Eu achei que as várias funções do sistema estão bem integradas.
6. Eu achei que há muita inconsistência no sistema.
7. Eu acredito que as pessoas aprenderão a usar este sistema rapidamente.
8. Eu achei o sistema confuso de usar.
9. Eu me senti confiante ao usar o sistema.
10. Precisei aprender muitas coisas novas antes de conseguir usar o sistema.

&emsp;As respostas foram transformadas em pontuações normalizadas (0 a 100). A média geral obtida foi **74**, o que representa **bom nível de usabilidade segundo os critérios internacionais**.


