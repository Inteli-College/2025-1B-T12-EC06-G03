---
title: Interface do modedlo
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Interface do Modelo
&emsp;Durante a segunda sprint, foi desenvolvida uma interface simplificada para permitir uma melhor visualização e interação com os modelos feitos para identificar e classificar as fissuras nas imagens. Assim, foi construída uma simples página WEB utilizando a ferramenta Streamlit, uma biblioteca que permite, em Python, criar dashboards interativas de maneira simples, acelerando o desenvolviento. Além disso, vale ressaltar que essa solução utilizando Streamlit é apenas provisória e que, ao final do projeto, espera-se ter essa interface integrada no próprio sistema da aplicação e não como um sistema a parte. 

## Funcionamento da interface
&emsp;Como dito anteriormente, a ideia de tal interface é ser a mais simples o possível para possibilitar apenas a interação com os modelos sem que o usuário final tenha que tocar em, por exemplo, uma janela de terminal. Assim, na interface gráfica construída com Streamlit, existem as seguintes funcionalidades:
* Visualizar informações básicas sobre os modelos desenvolvidos;
* Realizar o upload de uma ou mais imagens (é possível enviar várias imagens de uma vez) para os modelos analisarem;
* Visualizar a análise e classificação dos modelos para cada imagem, recebendo um resultado que diz se a fissura é de retração ou térmica. 

