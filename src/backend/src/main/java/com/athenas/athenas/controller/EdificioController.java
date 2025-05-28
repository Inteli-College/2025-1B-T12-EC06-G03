package com.athenas.athenas.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.service.EdificioService;

@RestController
@RequestMapping("/api/edificio")
public class EdificioController {

    private final EdificioService edificioService;
    private final ProjetoRepository projetoRepository;


    public EdificioController(EdificioService edificioService, ProjetoRepository projetoRepository) {
        this.projetoRepository = projetoRepository;
        this.edificioService = edificioService;
    }

    @GetMapping("/{projetoId}/edificios")
    public List<Edificio> getAllEdificios(
        @PathVariable Long projetoId
    ) {
        Projeto projeto = projetoRepository.findById(projetoId).orElse( null);
        return edificioService.getAllEdificiosFromProject(projeto);
    }
    
}
