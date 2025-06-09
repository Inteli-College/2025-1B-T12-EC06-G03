package com.athenas.athenas.controller;

import com.athenas.athenas.model.Fissura;
import com.athenas.athenas.repository.FissuraRepository;
import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.repository.ImagemRepository;
import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.repository.FachadaRepository;
import com.athenas.athenas.dto.FissuraPorcentagemDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.service.FissuraService;
import com.athenas.athenas.dto.FissuraDetalheDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/fissura")
public class FissuraController {

    @Autowired
    private FissuraService fissuraService;

    @Autowired
    private ProjetoRepository projetoRepository;
    @Autowired
    private EdificioRepository edificioRepository;
    @Autowired
    private FachadaRepository fachadaRepository;
    @Autowired
    private ImagemRepository imagemRepository;
    @Autowired
    private FissuraRepository fissuraRepository;

    @PostMapping
    public ResponseEntity<Fissura> createFissura(@RequestBody Fissura fissura) {
        try {
            // Validar se a imagem existe
            if (fissura.getImagem() == null || fissura.getImagem().getId() == null) {
                return ResponseEntity.badRequest().build();
            }
            
            Imagem imagem = imagemRepository.findById(fissura.getImagem().getId())
                .orElseThrow(() -> new RuntimeException("Imagem não encontrada"));
            
            // Configurar a imagem completa no objeto fissura
            fissura.setImagem(imagem);
            
            // Definir data de detecção se não foi fornecida
            if (fissura.getDataDeteccao() == null) {
                fissura.setDataDeteccao(java.time.LocalDateTime.now());
            }
            
            Fissura savedFissura = fissuraRepository.save(fissura);
            return ResponseEntity.status(HttpStatus.CREATED).body(savedFissura);
            
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @PutMapping("/{id}/aprovar")
    public ResponseEntity<Fissura> aprovarFissura(@PathVariable Long id, @RequestBody Map<String, Object> request) {
        try {
            Fissura fissura = fissuraRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Fissura não encontrada"));
            
            fissura.setAprovado(true);
            fissura.setAprovadoPor((String) request.get("aprovadoPor"));
            
            Fissura updatedFissura = fissuraRepository.save(fissura);
            return ResponseEntity.ok(updatedFissura);
            
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    @GetMapping("/porcentagem/{projetoId}")
    public FissuraPorcentagemDTO getPorcentagemPorTipo(@PathVariable Integer projetoId) {
        return fissuraService.getPorcentagemPorTipo(projetoId);
    }

    @GetMapping("/projeto/{projetoId}")
    public List<Fissura> getFissurasByProjeto(@PathVariable Integer projetoId) {
        Projeto projeto = projetoRepository.findById(projetoId)
            .orElseThrow(() -> new RuntimeException("Projeto not found with id: " + projetoId));
        List<Edificio> edificios = edificioRepository.findByProjeto(projeto);

        List<Fachada> fachadas = new ArrayList<>();
        for (Edificio edificio : edificios) {
            fachadas.addAll(fachadaRepository.findByEdificio(edificio));
        }

        List<Imagem> imagens = new ArrayList<>();
        for (Fachada fachada : fachadas) {
            imagens.addAll(imagemRepository.findByFachada(fachada));
        }

        List<Fissura> fissuras = new ArrayList<>();
        for (Imagem img : imagens) {
            fissuras.addAll(fissuraRepository.findByImagem(img));
        }

        return fissuras;
    }

    @GetMapping("/detalhes/projeto/{projetoId}")
    public List<FissuraDetalheDTO> getFissurasDetalhadasByProjeto(@PathVariable Integer projetoId) {
        Projeto projeto = projetoRepository.findById(projetoId)
            .orElseThrow(() -> new RuntimeException("Projeto not found with id: " + projetoId));
        List<Edificio> edificios = edificioRepository.findByProjeto(projeto);

        List<Fachada> fachadas = new ArrayList<>();
        for (Edificio edificio : edificios) {
            fachadas.addAll(fachadaRepository.findByEdificio(edificio));
        }

        List<Imagem> imagens = new ArrayList<>();
        for (Fachada fachada : fachadas) {
            imagens.addAll(imagemRepository.findByFachada(fachada));
        }

        List<FissuraDetalheDTO> detalhes = new ArrayList<>();
        for (Imagem img : imagens) {
            List<Fissura> fissuras = fissuraRepository.findByImagem(img);
            for (Fissura fissura : fissuras) {
                FissuraDetalheDTO dto = new FissuraDetalheDTO();
                dto.setId(fissura.getId());
                dto.setTipo(fissura.getTipo());
                dto.setCoordenadas(fissura.getCoordenadas());
                dto.setGravidade(fissura.getGravidade());
                dto.setDataDeteccao(fissura.getDataDeteccao() != null ? fissura.getDataDeteccao().toString() : null);
                dto.setConfianca(fissura.getConfianca());
                dto.setNomeImagem(img.getCaminhoArquivo());
                dto.setAprovado(fissura.getAprovado());
                dto.setAprovadoPor(fissura.getAprovadoPor());
                detalhes.add(dto);
            }
        }
        return detalhes;
    }

    @GetMapping("/imagem/{imagemId}/detalhes")
    public ResponseEntity<List<FissuraDetalheDTO>> getFissurasDetalhesByImagem(@PathVariable Long imagemId) {
        Imagem imagem = imagemRepository.findById(imagemId)
            .orElseThrow(() -> new RuntimeException("Imagem not found with id: " + imagemId));

        List<Fissura> fissuras = fissuraRepository.findByImagem(imagem);

        List<FissuraDetalheDTO> detalhes = new ArrayList<>();
        for (Fissura fissura : fissuras) {
            FissuraDetalheDTO dto = new FissuraDetalheDTO();
            dto.setId(fissura.getId());
            dto.setTipo(fissura.getTipo());
            dto.setCoordenadas(fissura.getCoordenadas());
            dto.setGravidade(fissura.getGravidade());
            dto.setConfianca(fissura.getConfianca());
            dto.setNomeImagem(imagem.getCaminhoArquivo()); 
            detalhes.add(dto);
        }
        return ResponseEntity.ok(detalhes);
    }
}
