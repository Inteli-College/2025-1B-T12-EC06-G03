---
title: Planejamento dos testes
sidebar_position: 3
---

&nbsp;&nbsp;&nbsp;&nbsp;Para assegurar que o sistema atenda plenamente aos requisitos funcionais (RF) e não funcionais (RNF), é essencial a realização de uma série de testes. Sem essa validação, não podemos garantir que o sistema satisfaça as necessidades técnicas e de usabilidade do usuário final. Os testes a seguir são divididos em duas categorias principais: validação de métricas objetivas e validação dependente da interação com o usuário.


## Testes Objetivos de Validação de Métricas

&nbsp;&nbsp;&nbsp;&nbsp;No Quadro 1, detalhamos os Testes Objetivos de Validação de Métricas, um conjunto de avaliações que visam a quantificar o desempenho e a conformidade do sistema. Esses testes são projetados para medir resultados específicos e objetivos, garantindo que as funcionalidades essenciais e os requisitos não funcionais sejam atendidos de forma mensurável e verificável.

<div align="center">
<sup>Quadro 1 - Testes objetivos</sup>

| ID  | RF/RNF Relacionado | Descrição do Teste                                                  | Método                                                         | Métrica de Aceite                                                  |
| --- | ------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| T01 | RNF1               | Verificar qualidade da transmissão de vídeo       | Medição com OpenCV simulando diferentes redes                  | FPS ≥ 10; Latência ≤ 500 ms                                        |
| T02 | RNF2 / RF02        | Avaliar acurácia da IA na detecção de fissuras                      | Validação cruzada com datasets rotulados                       | Acurácia ≥ 70%, variação ≤ 5%                                      |
| T03 | RF03               | Verificar consistência e integridade dos relatórios automáticos     | Verificação do conteúdo dos relatórios gerados      | Relatório completo |
| T04 | RF04               | Testar o armazenamento e organização do histórico de detecções      | Execução de coletas em datas diferentes e análise do histórico | Histórico com datas e logs de alterações         |
| T05 | RF05               | Verificar precisão da localização das fissuras nas imagens | Testes com imagens conhecidas e validação geométrica           | 100% das fissuras detectadas corretamente sobrepostas              |
| T06 | RF08               | Testar integridade e segurança da base de dados                     | Testes de inserção, leitura e falhas; simulação de ataques     | Sem perdas de dados; confidencialidade garantida                   |
| T07 | RF11               | Verificar o registro correto de logs de atividades                  | Auditoria de logs após ações críticas                          | 100% das ações críticas devidamente registradas com hora e ID      |

<sup>Fonte: Material Produzido pelos autores. (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;A seguir, cada um dos testes apresentados no Quadro 1 é detalhado individualmente. Essas descrições expandem o escopo, os métodos e os critérios de aceitação, fornecendo uma compreensão aprofundada de como cada teste será executado para validar as métricas e garantir a qualidade e confiabilidade do sistema.

### T01 — Teste de Qualidade da Transmissão de Vídeo

&nbsp;&nbsp;&nbsp;&nbsp;Simular a recepção de imagens via drone em diferentes condições de rede para medir o FPS (frames por segundo) e a latência.
* **Como testar:**
    * Usar OpenCV para medir FPS e latência durante a transmissão ao vivo.
    * Realizar os testes em cenários de rede variados: Wi-Fi estável, Wi-Fi instável (simulando perda de pacotes ou alta latência) e dados móveis (3G/4G/5G) com diferentes níveis de sinal.
    * Gravar as métricas em cada cenário para análise.
* **Critério de Aceitação:**
    * O FPS deve ser maior ou igual a 10 em todas as condições testadas.
    * A latência deve ser menor ou igual a 500 ms em todas as condições testadas.

### T02 — Acurácia do Modelo de Detecção de Fissuras

&nbsp;&nbsp;&nbsp;&nbsp;Avaliar a acurácia do modelo de classificação na detecção automática de fissuras em imagens coletadas.
* **Como testar:**
  * Separar dataset rotulado com imagens reais.
  * Realizar validação cruzada.
  * Medir precisão, recall, F1-score e acurácia.
* **Critério de Aceitação:**
  * Acurácia ≥ 70%
  * Variação entre execuções ≤ 5%

### T03 — Verificação dos Relatórios Gerados

&nbsp;&nbsp;&nbsp;&nbsp;Validar se os relatórios gerados contêm todas as informações exigidas.
* **Como testar:**
  * Gerar múltiplos relatórios com dados simulados.
  * Analisar presença das imagens com suas respectivas classificações no relatório, além das demais informações inseridas, como edifícios e pessoas responsáveis.
* **Critério de Aceitação:**
  * Todos os campos obrigatórios presentes e corretos.

### T04 — Armazenamento e Consulta do Histórico

&nbsp;&nbsp;&nbsp;&nbsp;Verificar se o sistema mantém e exibe corretamente o histórico de detecções para uma edificação.
* **Como testar:**
  * Fazer três coletas em datas diferentes para a mesma edificação.
  * Acessar histórico e comparar resultados.
* **Critério de Aceitação:**
  * Histórico correto e navegável.

### T05 — Sobreposição Gráfica das Fissuras

&nbsp;&nbsp;&nbsp;&nbsp; Validar se as localizações das fissuras foram detectadas corretamente.
* **Como testar:**
  * Usar imagens de teste com fissuras previamente marcadas.
  * Comparar visualmente os resultados.
* **Critério de Aceitação:**
  * As fissuras devem estar selecionadas corretamente nas imagens.

### T06 — Segurança e Integridade da Base de Dados

&nbsp;&nbsp;&nbsp;&nbsp;Testar a integridade, disponibilidade e confidencialidade dos dados armazenados.
* **Como testar:**
  * Realizar 100 inserções simultâneas com dados distintos via múltiplas threads/processos.
  * Interromper a conexão com o banco no meio de uma operação de escrita.
  * Validar, após cada inserção, se os dados estão completos e corretos.
  * Tentar acessar dados confidenciais (senhas dos usuários).
* **Critério de Aceitação:**
  * Nenhum dado deve ser corrompido, duplicado ou perdido.
  * Os dados confidenciais não podem estar disponíveis.

### T07 — Registro de Logs

&nbsp;&nbsp;&nbsp;&nbsp;Verificar se todas as ações de alteração dos relatórios estão sendo corretamente registradas.
* **Como testar:**
  * Realizar operações de detecção, edição e geração de relatório.
  * Acessar log do sistema e confirmar os registros.
* **Critério de Aceitação:**
  * Todas as edições realizadas no relatório devem estar registradas com horário na parte de logs do relatório.

## Testes Dependentes de Validação do Usuário

&nbsp;&nbsp;&nbsp;&nbsp;O Quadro 2 a seguir apresenta os Testes Dependentes da Validação do Usuário. Diferente dos testes objetivos, estas avaliações são focadas na experiência e percepção do usuário final, garantindo que o sistema não apenas funcione tecnicamente, mas também atenda às expectativas de usabilidade, intuitividade e satisfação. Desse modo, eles buscam confirmar que a solução é prática e eficaz no contexto de uso real.

<div align="center">
<sup>Quadro 2 - Testes de validação do usuário</sup>

| ID  | RF Relacionado | Descrição do Teste                                                 | Método                                              | Critério de Aceitação                                            |
| --- | -------------- | ------------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------- |
| T08 | RF01           | Avaliar facilidade e confiabilidade da integração com dispositivos | Sessão prática com operadores reais                 | Operadores confirmam envio automático sem intervenção manual     |
| T09 | RF06           | Avaliação de usabilidade da interface gráfica                      | Testes com engenheiros e técnicos prediais          | Os usuários avaliam como “intuitiva” ou “muito intuitiva” |
| T10 | RF09           | Validação do sistema de autenticação e permissões                  | Testes com diferentes perfis (admin, operador)      | Acesso permitido e bloqueado corretamente conforme perfil        |
| T11 | RF10           | Teste da funcionalidade de validação manual de fissuras            | Operadores revisam, aceitam, ou descartam detecções | Interface permite edição com facilidade  |

<sup>Fonte: Material Produzido pelos autores. (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;A seguir, estão detalhados cada um dos testes listados no Quadro 2. Estas descrições aprofundam os métodos de execução e os critérios de sucesso para cada teste, focando na validação da experiência do usuário e na confirmação de que o sistema é intuitivo, eficaz e atende às expectativas de quem realmente o utilizará no dia a dia.

### T08 — Validação da Integração com Dispositivos
&nbsp;&nbsp;&nbsp;&nbsp;Verificar com usuários reais a facilidade e eficácia do envio automático de imagens via drones/câmeras.
* **Como testar:**
  * Operadores realizam coleta com o drone conectado ao sistema.
  * Usuário realiza upload de imagem.
  * Observa-se se as imagens estão no sistema.
* **Critério de Aceitação:**
  * Todas as imagens devem estar salvas no sistema.

### T09 — Usabilidade da Interface Gráfica
&nbsp;&nbsp;&nbsp;&nbsp;Avaliar se a interface gráfica é compreensível, funcional e eficiente para os usuários-alvo (engenheiros civis e técnicos), durante a execução dos principais fluxos do sistema.
* **Fluxos Avaliados:**
1. Cadastro de usuário
2. Cadastro de empresa
3. Criação de projeto
4. Acesso a projeto existente
5. Criação de edificação
6. Upload de imagens
7. Aprovação das imagens para envio ao modelo
8. Verificação da classificação automática realizada pelo modelo
9. Exportação de relatório final
* **Como testar (Para cada fluxo):**
  * Fornecer apenas o objetivo (ex: “Crie uma nova edificação dentro do projeto X”).
  * Monitorar o tempo necessário, cliques, erros e dúvidas durante o uso da interface.
  * O avaliador só deve intervir se o participante travar completamente.
  * Aplicar um questionário de avaliação da usabilidade com as seguintes perguntas [[1]](#referências), cujas respostas devem estar na escala :
    > 1. Eu acho que gostaria de usar esse sistema com frequência.
    > 2. Eu acho o sistema desnecessariamente complexo.
    > 3. Eu achei o sistema fácil de usar.
    > 4. Eu acho que precisaria de ajuda de uma pessoa com conhecimentos técnicos para usar o sistema.
    > 5. Eu acho que as várias funções do sistema estão muito bem integradas.
    > 6. Eu acho que o sistema apresenta muita inconsistência.
    > 7. Eu imagino que as pessoas aprenderão como usar esse sistema rapidamente.
    > 8. Eu achei o sistema atrapalhado de usar.
    > 9. Eu me senti confiante ao usar o sistema.
    > 10. Eu precisei aprender várias coisas novas antes de conseguir usar o sistema.
  * Calcule a pontuação: para as respostas ímpares (1, 3, 5), subtraia 1 da pontuação que o usuário respondeu; para as respostas pares (2 e 4), subtraia a resposta de 5; some todos os valores das dez perguntas e multiplique por 2.5.
* **Critério de Aceitação:**
  * Média de pontuação acima de 68, dado que a média do System Usability Score é 68 pontos.

### T11 — Permissões e Autenticação
&nbsp;&nbsp;&nbsp;&nbsp;Confirmar se o controle de acesso funciona corretamente para diferentes perfis de usuário.
* **Como testar:**
  * Testar ações de leitura, escrita e administração com contas de administrador e operador.
* **Critério de Aceitação:**
  * Apenas administradores têm acesso completo; operadores têm acesso restrito.

### T12 — Validação Manual das Fissuras
&nbsp;&nbsp;&nbsp;&nbsp;Verificar a funcionalidade de revisão, edição e exclusão de fissuras detectadas automaticamente.
* **Como testar:**
  * Usuário técnico acessa imagem com detecções e interage com elas (valida, edita, descarta).
* **Critério de Aceitação:**
  * Usuário consegue modificar/validar todas as detecções conforme desejado, sem erro.

&nbsp;&nbsp;&nbsp;&nbsp;A documentação completa dos testes realizados, incluindo os resultados detalhados, logs de execução e status de cada validação, está disponível na próxima seção deste documento. 

# Referências

[1] TEIXEIRA, F. O que é o SUS (System Usability Scale) e como usá-lo em seu site. Disponível em: https://brasil.uxdesign.cc/o-que-%C3%A9-o-sus-system-usability-scale-e-como-us%C3%A1-lo-em-seu-site-6d63224481c8.