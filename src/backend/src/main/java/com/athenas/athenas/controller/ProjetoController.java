package com.athenas.athenas.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.dto.ProjetoDTO;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.service.ProjetoService;

@RestController
@RequestMapping("/api/projetos")
public class ProjetoController {

    private final ProjetoService projetoService;

    public ProjetoController(ProjetoService projetoService) {
        this.projetoService = projetoService;
    }

    @GetMapping
    public List<Projeto> listAllProjects() {
        return projetoService.findAll();
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Projeto createProject(@RequestBody ProjetoDTO projetoDTO) {
        Projeto projeto = new Projeto();
        projeto.setNome(projetoDTO.getNome());
        projeto.setDescricao(projetoDTO.getDescricao());
        projeto.setStatus(projetoDTO.getStatus());

        return projetoService.saveWithEmpresaId(projeto, projetoDTO.getEmpresa());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Projeto> getProjectById(@PathVariable Long id) {
        return projetoService.findById(id)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}")
    public ResponseEntity<Projeto> updateProject(@PathVariable Long id, @RequestBody ProjetoDTO projeto) {
        if (!projetoService.findById(id).isPresent()) {
            return ResponseEntity.notFound().build();
        }
        Projeto existingProject = projetoService.findById(id).get();
        existingProject.setNome(projeto.getNome());
        existingProject.setDescricao(projeto.getDescricao());
        existingProject.setStatus(projeto.getStatus());

        return ResponseEntity.ok(projetoService.updateWithEmpresaId(existingProject, projeto.getEmpresa()));
    }
}
