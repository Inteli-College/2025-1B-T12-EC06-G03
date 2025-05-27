package com.athenas.athenas.service;

import com.athenas.athenas.DTO.FissuraPorcentagemDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Fissura;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.FachadaRepository;
import com.athenas.athenas.repository.ImagemRepository;
import com.athenas.athenas.repository.FissuraRepository;
import com.athenas.athenas.repository.ProjetoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class FissuraService {

    @Autowired
    private EdificioRepository edificioRepository;
    @Autowired
    private FachadaRepository fachadaRepository;
    @Autowired
    private ImagemRepository imagemRepository;
    @Autowired
    private FissuraRepository fissuraRepository;
    @Autowired
    private ProjetoRepository projetoRepository;

    public FissuraPorcentagemDTO getPorcentagemPorTipo(Integer projetoId) {
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

        Map<String, Long> contagem = fissuras.stream()
            .collect(Collectors.groupingBy(Fissura::getTipo, Collectors.counting()));

        int total = fissuras.size();
        Map<String, Integer> porcentagem = new HashMap<>();
        for (Map.Entry<String, Long> entry : contagem.entrySet()) {
            porcentagem.put(entry.getKey(), total > 0 ? (int) Math.round((entry.getValue() * 100.0) / total) : 0);
        }

        FissuraPorcentagemDTO dto = new FissuraPorcentagemDTO();
        dto.setPorcentagemPorTipo(porcentagem);
        return dto;
    }
}