package com.athenas.athenas.service;

import java.util.Arrays;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.athenas.athenas.DTO.ViewProjetoResponseDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.ProjetoRepository;

@Service
public class ViewProjetoService {

    @Autowired
    private ProjetoRepository projetoRepository;

    public ViewProjetoResponseDTO ReadProjectData(int idProjeto) {
        ViewProjetoResponseDTO response = new ViewProjetoResponseDTO();

        Projeto projeto = projetoRepository.findById(idProjeto)
            .orElseThrow(() -> new RuntimeException("Project not found with id: " + idProjeto));
        
        response.setNome(projeto.getNome());
        response.setResponsaveisNomes(Arrays.asList("Maria Lima", "Rafael Silva"));
        response.setEmpresaNome("USP");
        
        Edificio edificio = new Edificio();
        edificio.setNome("Prédio do LMPC Escola Politécnica da USP");
        edificio.setLocalizacao("Av. Professor Luciano Gualberto, travessa 3, n.º 158, São Paulo – SP");
        edificio.setTipo("Pesquisa e Ensino");
        edificio.setPavimentos(2);
        
        response.setEdificios(Arrays.asList(edificio));
        
        response.setDescricao("Este projeto tem como objetivo identificar fissuras na estrutura do prédio do LMPC, localizado na Escola Politécnica da USP. Utilizando imagens capturadas por drone, o sistema analisa as fachadas do edifício para detectar possíveis falhas estruturais.");
        
        response.setLogs(Arrays.asList(
            "06/05/2025 - Upload da Imagem Captura01.png",
            "05/05/2025 - Análise da Imagem Upload03.png feita"
        ));
        
        return response;
    }
}