package com.athenas.athenas.controller;

import java.util.List;
import java.util.Optional;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.service.EdificioService;

@RestController
@RequestMapping("/api/edificio")
public class EdificioController {
    
    private final EdificioRepository edificioRepository;
    private final ProjetoRepository projetoRepository;

    public EdificioController(EdificioService edificioService, 
                            EdificioRepository edificioRepository,
                            ProjetoRepository projetoRepository) {
        this.edificioRepository = edificioRepository;
        this.projetoRepository = projetoRepository;
    }

    @PostMapping
    public ResponseEntity<Edificio> createEdificio(@RequestBody Edificio edificio) {
        try {
            Edificio savedEdificio = edificioRepository.save(edificio);
            return new ResponseEntity<>(savedEdificio, HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/{projetoId}")
    public ResponseEntity<Edificio> createEdificioForProject(
            @PathVariable Long projetoId,
            @RequestBody Edificio edificio) {
        try {
            Optional<Projeto> projetoOpt = projetoRepository.findById(projetoId);
            if (!projetoOpt.isPresent()) {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
            
            edificio.setProjeto(projetoOpt.get());
            Edificio savedEdificio = edificioRepository.save(edificio);
            return new ResponseEntity<>(savedEdificio, HttpStatus.CREATED);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping
    public ResponseEntity<List<Edificio>> getAllEdificios() {
        try {
            List<Edificio> edificios = edificioRepository.findAll();
            if (edificios.isEmpty()) {
                return new ResponseEntity<>(HttpStatus.NO_CONTENT);
            }
            return new ResponseEntity<>(edificios, HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<Edificio> getEdificioById(@PathVariable Long id) {
        try {
            Optional<Edificio> edificio = edificioRepository.findById(id);
            if (edificio.isPresent()) {
                return new ResponseEntity<>(edificio.get(), HttpStatus.OK);
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/projeto/{projetoId}")
    public ResponseEntity<List<Edificio>> getAllEdificiosByProject(@PathVariable Long projetoId) {
        try {
            Optional<Projeto> projetoOpt = projetoRepository.findById(projetoId);
            if (!projetoOpt.isPresent()) {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
            
            List<Edificio> edificios = edificioRepository.findByProjeto(projetoOpt.get());
            if (edificios.isEmpty()) {
                return new ResponseEntity<>(HttpStatus.NO_CONTENT);
            }
            return new ResponseEntity<>(edificios, HttpStatus.OK);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/{id}")
    public ResponseEntity<Edificio> updateEdificio(@PathVariable Long id, @RequestBody Edificio edificio) {
        try {
            Optional<Edificio> existingEdificio = edificioRepository.findById(id);
            if (existingEdificio.isPresent()) {
                edificio.setId(id);
                Edificio updatedEdificio = edificioRepository.save(edificio);
                return new ResponseEntity<>(updatedEdificio, HttpStatus.OK);
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<HttpStatus> deleteEdificio(@PathVariable Long id) {
        try {
            Optional<Edificio> edificio = edificioRepository.findById(id);
            if (edificio.isPresent()) {
                edificioRepository.deleteById(id);
                return new ResponseEntity<>(HttpStatus.NO_CONTENT);
            } else {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @DeleteMapping("/projeto/{projetoId}")
    public ResponseEntity<HttpStatus> deleteAllEdificiosByProject(@PathVariable Long projetoId) {
        try {
            Optional<Projeto> projetoOpt = projetoRepository.findById(projetoId);
            if (!projetoOpt.isPresent()) {
                return new ResponseEntity<>(HttpStatus.NOT_FOUND);
            }
            
            List<Edificio> edificios = edificioRepository.findByProjeto(projetoOpt.get());
            edificioRepository.deleteAll(edificios);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

}