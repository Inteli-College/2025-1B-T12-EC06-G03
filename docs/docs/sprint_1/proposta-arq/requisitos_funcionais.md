---
title: Requisitos Funcionais
sidebar_position: 1
---

# Requisitos Funcionais

##### Requisitos funcionais do projeto

&emsp;Os requisitos funcionais descrevem as funcionalidades essenciais que o sistema deve oferecer para garantir a identificação automatizada, o monitoramento contínuo e a análise preditiva de fissuras em edificações, com foco em revestimentos de argamassa. Esses requisitos estabelecem as interações do sistema com os usuários e com plataformas de captura de imagens, como drones e câmeras de alta resolução, detalhando as capacidades necessárias para processar imagens, detectar fissuras com inteligência artificial, classificar sua gravidade, registrar seu histórico de evolução e emitir alertas preventivos. Além disso, abrangem funcionalidades como geração de relatórios automáticos, visualização gráfica das fissuras sobre as imagens originais, validação manual por engenheiros, e registro de atividades para fins de rastreabilidade e auditoria. Dessa forma, esses requisitos garantem que o sistema atenda aos objetivos de precisão técnica, usabilidade e confiabilidade, contribuindo significativamente para a manutenção preditiva, a segurança estrutural e a redução de custos operacionais nas construções monitoradas, como é possível observar na tabela a seguir:

### Tabela de Requisitos Funcionais 

| RF#  | Descrição | Regra de negócio |
|------|-----------|------------------|
| RF01 | O sistema deve integrar-se com drones ou câmeras de alta resolução para receber imagens automaticamente. | A integração deve permitir o envio automatizado de imagens capturadas em campo, reduzindo a necessidade de intervenção manual e aumentando a agilidade do processo. |
| RF02 | O sistema deve detectar automaticamente fissuras nas imagens recebidas. | A detecção deve ocorrer com base em técnicas de visão computacional e IA treinadas para identificar padrões típicos de fissuras em revestimentos. |
| RF03 | O sistema deve classificar a gravidade das fissuras identificadas. | As fissuras devem ser categorizadas em níveis (leve, moderada, severa) com base em características como comprimento, largura e forma. |
| RF04 | O sistema deve gerar relatórios automáticos com as fissuras detectadas. | Cada relatório deve conter informações como data, localização da fissura, gravidade e recomendações de manutenção. |
| RF05 | O sistema deve manter um histórico de detecções para monitoramento da evolução das fissuras. | O histórico deve ser armazenado por edificação e permitir comparações temporais entre diferentes coletas. |
| RF06 | O sistema deve emitir alertas quando forem detectadas fissuras classificadas como críticas. | O alerta deve ser exibido na interface e enviado por e-mail ou notificação a usuários cadastrados, quando aplicável. |
| RF07 | O sistema deve permitir a visualização gráfica das fissuras detectadas sobrepostas às imagens originais. | O usuário deve conseguir identificar a localização exata de cada fissura de forma intuitiva. |
| RF08 | O sistema deve permitir a geração de relatórios customizados com base em filtros definidos pelo usuário. | O usuário poderá escolher o período, local da edificação e nível de gravidade das fissuras para gerar relatórios específicos. |
| RF09 | O sistema deve possuir uma interface gráfica responsiva e acessível via desktop. | A interface deve ser intuitiva e acessível para engenheiros civis e técnicos de manutenção predial. |
| RF10 | O sistema deve possibilitar o cadastro de edificações e áreas monitoradas. | Cada edificação deve conter seus próprios registros e imagens, organizados por data e localização. |
| RF11 | O sistema deve armazenar as imagens e dados analisados em uma base de dados segura. | A base de dados deve permitir acesso rápido e garantir integridade e confidencialidade das informações. |
| RF12 | O sistema deve permitir autenticação de usuários com diferentes níveis de permissão. | Usuários administrativos terão acesso total, enquanto operadores terão acesso restrito às funcionalidades básicas. |
| RF13 | O sistema deve prever a evolução das fissuras ao longo do tempo. | O modelo preditivo deve utilizar dados históricos para estimar o crescimento e agravamento das fissuras, permitindo decisões de manutenção mais estratégicas. |
| RF14 | O sistema deve permitir a validação manual de fissuras detectadas automaticamente. | O usuário técnico poderá revisar, editar ou descartar fissuras identificadas pela IA, garantindo maior confiabilidade na análise. |
| RF15 | O sistema deve registrar logs de atividades dos usuários. | Todas as ações críticas, como detecção, edição, geração de relatórios ou exclusão de dados, devem ser registradas com data/hora e identificação do responsável. |

&emsp;Os requisitos funcionais definidos para o sistema de detecção e monitoramento automatizado de fissuras em edificações garantem que a solução atenda de forma precisa às demandas técnicas e operacionais da construção civil. A implementação dessas funcionalidades permitirá a análise inteligente de imagens capturadas por drones, a detecção automatizada de fissuras, a classificação de sua gravidade, o acompanhamento de sua evolução ao longo do tempo e a emissão de alertas preditivos para manutenção. Além disso, recursos como visualização gráfica, geração de relatórios e validação manual proporcionarão maior controle e confiabilidade aos profissionais envolvidos na inspeção e conservação das estruturas. Com isso, o sistema contribuirá significativamente para a prevenção de danos estruturais, a redução de custos com manutenção corretiva e o aumento da segurança nas edificações monitoradas, consolidando-se como uma ferramenta inovadora e estratégica para o setor.

---

## Bibliografia

FIGUEIREDO, Eduardo. Requisitos funcionais e requisitos não funcionais. Icex, Dcc/Ufmg, 2011. Acesso em: 10 de fevereiro de 2025

GASTALDO, Denise Lazzeri; MIDORIKAWA, Edson Toshimi. Processo de Engenharia de Requisitos Aplicado a Requisitos Não-Funcionais de Desempenho–Um Estudo de Caso. In: Workshop em Engenharia de Requisitos. Piracicaba. 2003. p. 302-316.
