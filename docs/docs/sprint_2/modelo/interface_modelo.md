---
title: Interface do modelo
sidebar_position: 2
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Interface do Modelo
&emsp;Durante a segunda sprint, foi desenvolvida uma interface simplificada para permitir uma melhor visualização e interação com os modelos feitos para identificar e classificar as fissuras nas imagens. Assim, foi construída uma simples página WEB utilizando a ferramenta Streamlit, uma biblioteca que permite, em Python, criar dashboards interativas de maneira simples, acelerando o desenvolvimento. Além disso, vale ressaltar que essa solução utilizando Streamlit é apenas provisória e que, ao final do projeto, espera-se ter essa interface integrada no próprio sistema da aplicação e não como um sistema à parte. 

## Funcionamento da interface
&emsp;Como dito anteriormente, a ideia de tal interface é ser a mais simples o possível para possibilitar apenas a interação com os modelos sem que o usuário final tenha que tocar em, por exemplo, uma janela de terminal. Assim, na interface gráfica construída com Streamlit, existem as seguintes funcionalidades:
* Visualizar informações básicas sobre os modelos desenvolvidos;
* Realizar o upload de uma ou mais imagens (é possível enviar várias imagens de uma vez) para os modelos analisarem;
* Visualizar a análise e classificação dos modelos para cada imagem, recebendo um resultado que diz se a fissura é de retração ou térmica. 

## Análise de fissuras
&emsp;Na interface gráfica feita com Streamlit, o processo de analisar o tipo de uma fissura presente numa imagem é tão simples quanto realizar o upload do arquivo da imagem. Abaixo (figura 1), é possível ver a tela que permite ao usuário enviar quantas imagens quiser. Ao clicar em "Browse files", o explorador de arquivos do usuário é aberto e ele pode selecionar as imagens que deseja analisar, desde que estejam em formato adequado (PNG, JPG ou JPEG) e não passe do tamanho limite de 200mb, embora este último possa ser alterado. 

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 1 - Upload de imagens na interface</strong></p>
  <img 
    src={useBaseUrl('/img/upload_streamlit.jpeg')} 
    alt="Upload de imagens na interface" 
    title="Upload de imagens na interface" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;Com uma imagem enviada para a plataforma, os modelos começam a trabalhar automaticamente para identificar a localização da fissura e também a sua classificação. Assim, logo após enviar a imagem, o usuário já consegue ver a classificação da imagem e sua anotação (figura 2).

<div style={{ textAlign: 'center' }}>
  <p><strong>Figura 2 - Classificação de imagens</strong></p>
  <img 
    src={useBaseUrl('/img/imagem_classificada.png')} 
    alt="Classificação de imagens" 
    title="Classificação de imagens" 
    style={{ maxWidth: '100%', height: 'auto' }}
  />
  <p>Fonte: Elaborado pelos autores (2025)</p>
</div>

&emsp;Na imagem acima (figura 2), foi enviada uma imagem de uma fissura de retração, que foi corretamente classificada pelo modelo como tal. É importante ressaltar que a imagem que foi enviada para o modelo analisar não estava presente no conjunto de treinamento dos modelos preditivos, ou seja, os modelos nunca a haviam visto anteriormente. 

## Conclusão
&emsp;Concluindo, foi feita uma interface básica com a biblioteca Streamlit, tendo como principal finalidade facilitar o uso dos modelos por usuários não técnicos. Embora essa versão da interface gráfica provavelmente não estará presente na versão final do projeto, ela é importante pois permite testar o funcionamento dos modelos de modo facilitado e também os mostrar para, por exemplo, os stakeholders do projeto neste momento de fase inicial de desenvolvimento.