package com.athenas.athenas.controller;

import com.athenas.athenas.model.Fissura;
import com.athenas.athenas.repository.FissuraRepository;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/fissuras")
public class FissuraController {

    private final FissuraRepository fissuraRepository;

    public FissuraController(FissuraRepository fissuraRepository) {
        this.fissuraRepository = fissuraRepository;
    }

    @GetMapping
    public List<Fissura> listarTodas() {
        return fissuraRepository.findAll();
    }

    @PostMapping("/{id}/aprovar")
    public Fissura aprovar(@PathVariable Long id, @RequestBody AprovarRequest body) {
        Fissura fissura = fissuraRepository.findById(id).orElseThrow();
        fissura.setAprovado(body.isAprovado());
        fissura.setAprovadoPor(body.getAprovadoPor());
        return fissuraRepository.save(fissura);
    }

    public static class AprovarRequest {
        private boolean aprovado;
        private String aprovadoPor;

        public boolean isAprovado() {
            return aprovado;
        }

        public void setAprovado(boolean aprovado) {
            this.aprovado = aprovado;
        }

        public String getAprovadoPor() {
            return aprovadoPor;
        }

        public void setAprovadoPor(String aprovadoPor) {
            this.aprovadoPor = aprovadoPor;
        }
    }
}
