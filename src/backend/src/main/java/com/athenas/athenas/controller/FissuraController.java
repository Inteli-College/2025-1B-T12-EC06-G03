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
import java.util.Optional;

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

    @PostMapping
    public ResponseEntity<Fissura> criarFissura(@RequestBody CriarFissuraRequest request) {
        try {
            // Buscar a imagem
            Imagem imagem = imagemRepository.findById(request.getImagem_id())
                .orElseThrow(() -> new RuntimeException("Imagem not found with id: " + request.getImagem_id()));
            
            // Criar nova fissura
            Fissura fissura = new Fissura();
            fissura.setImagem(imagem);
            fissura.setTipo(request.getTipo());
            fissura.setCoordenadas(request.getCoordenadas());
            fissura.setGravidade(request.getGravidade());
            fissura.setDataDeteccao(request.getData_deteccao());
            fissura.setConfianca(request.getConfianca());
            fissura.setAprovado(false); // Por padrão, não aprovado
            
            Fissura fissuraSalva = fissuraRepository.save(fissura);
            return ResponseEntity.status(HttpStatus.CREATED).body(fissuraSalva);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PutMapping("/{fissuraId}/aprovar")
    public ResponseEntity<Fissura> aprovarFissura(@PathVariable Long fissuraId, @RequestBody AprovarFissuraRequest request) {
        try {
            System.out.println("Tentando aprovar fissura ID: " + fissuraId);
            System.out.println("Request: aprovado=" + request.getAprovado() + ", aprovadoPor=" + request.getAprovadoPor());
            
            Optional<Fissura> fissuraOpt = fissuraRepository.findById(fissuraId);
            
            if (fissuraOpt.isEmpty()) {
                System.out.println("Fissura não encontrada com ID: " + fissuraId);
                return ResponseEntity.notFound().build();
            }
            
            Fissura fissura = fissuraOpt.get();
            fissura.setAprovado(request.getAprovado());
            fissura.setAprovadoPor(request.getAprovadoPor());
            
            System.out.println("Salvando fissura aprovada...");
            Fissura fissuraAtualizada = fissuraRepository.save(fissura);
            System.out.println("Fissura salva com sucesso: " + fissuraAtualizada.getId());
            
            return ResponseEntity.ok(fissuraAtualizada);
        } catch (Exception e) {
            System.err.println("Erro ao aprovar fissura: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    // DTO para o request de criação de fissura
    public static class CriarFissuraRequest {
        private Long imagem_id;
        private String tipo;
        private String coordenadas;
        private String gravidade;
        private java.time.LocalDateTime data_deteccao;
        private Double confianca;
        
        // Getters e setters
        public Long getImagem_id() { return imagem_id; }
        public void setImagem_id(Long imagem_id) { this.imagem_id = imagem_id; }
        
        public String getTipo() { return tipo; }
        public void setTipo(String tipo) { this.tipo = tipo; }
        
        public String getCoordenadas() { return coordenadas; }
        public void setCoordenadas(String coordenadas) { this.coordenadas = coordenadas; }
        
        public String getGravidade() { return gravidade; }
        public void setGravidade(String gravidade) { this.gravidade = gravidade; }
        
        public java.time.LocalDateTime getData_deteccao() { return data_deteccao; }
        public void setData_deteccao(java.time.LocalDateTime data_deteccao) { this.data_deteccao = data_deteccao; }
        
        public Double getConfianca() { return confianca; }
        public void setConfianca(Double confianca) { this.confianca = confianca; }
    }

    // DTO para o request de aprovação
    public static class AprovarFissuraRequest {
        private Boolean aprovado;
        private String aprovadoPor;
        
        public Boolean getAprovado() {
            return aprovado;
        }
        
        public void setAprovado(Boolean aprovado) {
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
