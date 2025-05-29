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

import java.util.ArrayList;
import java.util.List;

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
                dto.setNomeImagem(img.getCaminhoArquivo()); // <-- CORRIGIDO AQUI
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
