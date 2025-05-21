// json esperado:
// data = {
//     "projeto": "USP",
//     "responsaveis": ["Maria Lima", "Rafael Silva"],
//     "empresa": "USP",
//     "edificios": [{
//         "nome": "Prédio do LMPC Escola Politécnica da USP",
//         "localizacao": "Av. Professor Luciano Gualberto, travessa 3, n.º 158, São Paulo – SP",
//         "tipo": "Pesquisa e Ensino",
//         "pavimentos": 2,
//         "ano_construcao": "Estimado em 1980",
//     }],
//     "descricao": "Este projeto tem como objetivo  identificar fissuras na estrutura do prédio do LMPC, localizado na Escola Politécnica da USP. Utilizando imagens capturadas por drone, o sistema analisa as fachadas do edifício para detectar possíveis falhas estruturais.",
//     "logs_alteracoes": [
//         "06/05/2025 - Upload da Imagem Captura01.png",
//         "05/05/2025 - Análise da Imagem Upload03.png feita"
//     ]
// }

package com.athenas.athenas.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.DTO.ViewProjetoRequestDTO;
import com.athenas.athenas.DTO.ViewProjetoResponseDTO;
import com.athenas.athenas.service.ViewProjetoService;

@RestController
@RequestMapping("/api/projeto")
public class ViewProjetoController {

    private final ViewProjetoService viewProjetoService;

    @Autowired
    public ViewProjetoController(ViewProjetoService viewProjetoService) {
        this.viewProjetoService = viewProjetoService;
    }

    @PostMapping("/ViewProjeto")
    public ResponseEntity<ViewProjetoResponseDTO> viewProjeto(@RequestBody ViewProjetoRequestDTO viewProjetoRequestDTO) {
        try {
            ViewProjetoResponseDTO response = this.viewProjetoService.ReadProjectData(viewProjetoRequestDTO.getIdProjeto());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }
}
