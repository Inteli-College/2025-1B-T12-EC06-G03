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
| T03 | RF03               | Verificar consistência e integridade dos relatórios automáticos     | Verificação automática do conteúdo dos relatórios gerados      | Relatório completo com data, localização, gravidade, recomendações |
| T04 | RF04               | Testar o armazenamento e organização do histórico de detecções      | Execução de coletas em datas diferentes e análise do histórico | Histórico com versionamento por data e comparação temporal         |
| T05 | RF05               | Verificar precisão da localização das fissuras nas imagens | Testes com imagens conhecidas e validação geométrica           | 100% das fissuras detectadas corretamente sobrepostas              |
| T06 | RF08               | Testar integridade e segurança da base de dados                     | Testes de inserção, leitura e falhas; simulação de ataques     | Sem perdas de dados; confidencialidade garantida                   |
| T07 | RF11               | Verificar o registro correto de logs de atividades                  | Auditoria de logs após ações críticas                          | 100% das ações críticas devidamente registradas com hora e ID      |

<sup>Fonte: Material Produzido pelos autores. (2025)</sup>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;A seguir, cada um dos testes apresentados no Quadro 1 é detalhado individualmente. Essas descrições expandem o escopo, os métodos e os critérios de aceitação, fornecendo uma compreensão aprofundada de como cada teste será executado para validar as métricas e garantir a qualidade e confiabilidade do sistema.

### T01 — Teste de Qualidade da Transmissão de Vídeo

&nbsp;&nbsp;&nbsp;&nbsp;Simular a recepção de imagens via drone em diferentes condições de rede e medir o FPS (frames por segundo) e a latência.
* **Como testar:**
    * Usar OpenCV para medir FPS e latência durante a transmissão ao vivo.
    * Realizar os testes em cenários de rede variados: Wi-Fi estável, Wi-Fi instável (simulando perda de pacotes ou alta latência) e dados móveis (3G/4G/5G) com diferentes níveis de sinal.
    * Gravar as métricas em cada cenário para análise.
* **Critério de Aceitação:**
    * O **FPS** deve ser $\ge$ 10 em todas as condições testadas.
    * A **latência** deve ser $\le$ 500 ms em todas as condições testadas.

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

&nbsp;&nbsp;&nbsp;&nbsp;Validar se os relatórios automáticos gerados contêm todas as informações exigidas.
* **Como testar:**
  * Gerar múltiplos relatórios após simulações de detecção.
  * Analisar presença das imagens com suas respectivas clasificações no relatório.
* **Critério de Aceitação:**
  * Todos os campos obrigatórios presentes e corretos.

### T04 — Armazenamento e Consulta do Histórico

&nbsp;&nbsp;&nbsp;&nbsp;Verificar se o sistema mantém e exibe corretamente o histórico de detecções para uma edificação.
* **Como testar:**
  * Fazer três coletas em datas diferentes para a mesma edificação.
  * Acessar histórico e comparar resultados.
* **Critério de Aceitação:**
  * Histórico correto e navegável; registros por data visíveis.

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
  * Executar inserções simultâneas, falhas intencionais e tentativas de acesso indevido.
  * Validar recuperação e consistência dos dados.
* **Critério de Aceitação:**
  * Nenhum dado perdido ou exposto.
  * Dados recuperáveis após falhas simuladas.

### T07 — Registro de Logs

&nbsp;&nbsp;&nbsp;&nbsp;Verificar se todas as ações de alteração dos relatórios estão sendo corretamente registradas.
* **Como testar:**
  * Realizar operações de detecção, edição e geração de relatório.
  * Acessar log do sistema e confirmar os registros.
* **Critério de Aceitação:**
  * 100% das ações relevantes com horário no log.

## Testes Dependentes de Validação do Usuário

&nbsp;&nbsp;&nbsp;&nbsp;

| ID  | RF Relacionado | Descrição do Teste                                                 | Método                                              | Critério de Aceitação                                            |
| --- | -------------- | ------------------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------- |
| T08 | RF01           | Avaliar facilidade e confiabilidade da integração com dispositivos | Sessão prática com operadores reais                 | Operadores confirmam envio automático sem intervenção manual     |
| T09 | RF06           | Avaliação de usabilidade da interface gráfica                      | Testes com engenheiros e técnicos prediais          | ≥ 80% dos usuários avaliam como “intuitiva” ou “muito intuitiva” |
| T10 | RF07           | Validação do processo de cadastro de edificações                   | Simulação com usuários reais                        | Cadastro completo em até 3 minutos, sem erros                    |
| T11 | RF09           | Validação do sistema de autenticação e permissões                  | Testes com diferentes perfis (admin, operador)      | Acesso permitido e bloqueado corretamente conforme perfil        |
| T12 | RF10           | Teste da funcionalidade de validação manual de fissuras            | Operadores revisam, aceitam, ou descartam detecções | Interface permite edição com fluidez; 90% dos testes sem falhas  |
