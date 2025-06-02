package com.athenas.athenas.service;

import java.util.List;

import org.springframework.stereotype.Service;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;

@Service
public class EdificioService {
    private final EdificioRepository edificioRepository;
    public EdificioService(EdificioRepository edificioRepository) {
        this.edificioRepository = edificioRepository;
    }

    public List<Edificio> getAllEdificiosFromProject(Projeto projeto) {
        return edificioRepository.findByProjeto(projeto);
    }

}
