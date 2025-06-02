---
title: Interface do modelo
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Interface intermediária do Modelo
&emsp;Durante a terceira sprint, assim como na Sprint 2, foi desenvolvida uma interface simplificada para permitir uma melhor visualização e interação com o modelo de identificar e segmentação as fissuras nas imagens. Assim, foi construída uma simples página WEB utilizando a ferramenta Streamlit, uma biblioteca que permite, em Python, criar dashboards interativas de maneira simples, acelerando o desenvolvimento. Além disso, vale ressaltar que essa solução utilizando Streamlit é apenas provisória e que, ao final do projeto, espera-se ter essa interface integrada no próprio sistema da aplicação e não como um sistema à parte. 

## Funcionamento da interface
&emsp;Como dito anteriormente, a ideia de tal interface é ser a mais simples o possível para possibilitar apenas a interação com os modelos sem que o usuário final tenha que tocar em, por exemplo, uma janela de terminal. Assim, na interface gráfica construída com Streamlit, existem as seguintes funcionalidades:
* Visualizar informações básicas sobre os modelos desenvolvidos;
* Realizar o upload de uma ou mais imagens (é possível enviar várias imagens de uma vez) para os modelos analisarem;
* Visualizar a classificação e segmentação do modelo para cada imagem, recebendo um resultado que diz se a fissura é de retração ou térmica. 

## Análise de fissuras
&emsp;Na interface gráfica feita com Streamlit, o processo de analisar o tipo de uma fissura presente numa imagem é tão simples quanto realizar o upload do arquivo da imagem. Ao clicar em "Browse files", o explorador de arquivos do usuário é aberto e ele pode selecionar as imagens que deseja analisar, desde que estejam em formato adequado (PNG, JPG ou JPEG) e não passe do tamanho limite de 200mb, embora este último possa ser alterado. 

&emsp;Com uma imagem enviada para a plataforma, os modelos começam a trabalhar automaticamente para identificar a localização e identificação dos polígonos da fissura, além da sua segmentação. Assim, logo após enviar a imagem, o usuário já consegue ver a segmentação e classificação da imagem e sua anotação.

## Conclusão
&emsp;Foi elaborada uma interface básica com a biblioteca Streamlit, tendo como principal finalidade facilitar o uso dos modelos por usuários não técnicos. Embora essa versão da interface gráfica provavelmente não estará presente na versão final do projeto, ela é importante pois permite testar o funcionamento dos modelos de modo facilitado e também os mostrar para, por exemplo, os stakeholders do projeto neste momento de fase inicial de desenvolvimento.