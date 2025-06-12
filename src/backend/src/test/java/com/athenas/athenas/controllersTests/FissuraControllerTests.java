package com.athenas.athenas.controllersTests;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyInt;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.athenas.athenas.controller.FissuraController;
import com.athenas.athenas.dto.FissuraPorcentagemDTO;
import com.athenas.athenas.dto.FissuraDetalheDTO;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.model.Fissura;
import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.FachadaRepository;
import com.athenas.athenas.repository.FissuraRepository;
import com.athenas.athenas.repository.ImagemRepository;
import com.athenas.athenas.repository.ProjetoRepository;
import com.athenas.athenas.service.FissuraService;

@ExtendWith(MockitoExtension.class)
public class FissuraControllerTests {

    @Mock
    private FissuraService fissuraService;

    @Mock
    private ProjetoRepository projetoRepository;

    @Mock
    private EdificioRepository edificioRepository;

    @Mock
    private FachadaRepository fachadaRepository;

    @Mock
    private ImagemRepository imagemRepository;

    @Mock
    private FissuraRepository fissuraRepository;

    @InjectMocks
    private FissuraController fissuraController;

    private Fissura fissura;
    private Imagem imagem;
    private Fachada fachada;
    private Edificio edificio;
    private Projeto projeto;
    private FissuraPorcentagemDTO porcentagemDTO;
    private FissuraDetalheDTO detalheDTO;

    @BeforeEach
    void setUp() {
        projeto = new Projeto();
        projeto.setId(1L);
        projeto.setNome("Projeto Teste");

        edificio = new Edificio();
        edificio.setId(1L);
        edificio.setNome("Edificio Teste");
        edificio.setProjeto(projeto);

        fachada = new Fachada();
        fachada.setId(1L);
        fachada.setNome("Fachada Norte");
        fachada.setEdificio(edificio);

        imagem = new Imagem();
        imagem.setId(1L);
        imagem.setCaminhoArquivo("imagem_teste.jpg");
        imagem.setFachada(fachada);

        fissura = new Fissura();
        fissura.setId(1L);
        fissura.setTipo("VERTICAL");
        fissura.setCoordenadas("100,200,150,250");
        fissura.setGravidade("MEDIA");
        fissura.setConfianca(0.85);
        fissura.setDataDeteccao(LocalDateTime.now());
        fissura.setImagem(imagem);
        fissura.setAprovado(false);

        porcentagemDTO = new FissuraPorcentagemDTO();

        detalheDTO = new FissuraDetalheDTO();
        detalheDTO.setId(1L);
        detalheDTO.setTipo("VERTICAL");
        detalheDTO.setCoordenadas("100,200,150,250");
        detalheDTO.setGravidade("MEDIA");
        detalheDTO.setConfianca(0.85);
        detalheDTO.setNomeImagem("imagem_teste.jpg");
        detalheDTO.setAprovado(false);
    }

    // Testes para createFissura()
    @Test
    void testCreateFissura_Success() {
        when(imagemRepository.findById(1L)).thenReturn(Optional.of(imagem));
        when(fissuraRepository.save(any(Fissura.class))).thenReturn(fissura);

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissura);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals(fissura, response.getBody());
        verify(imagemRepository).findById(1L);
        verify(fissuraRepository).save(any(Fissura.class));
    }

    @Test
    void testCreateFissura_ImagemNull() {
        Fissura fissuraSemImagem = new Fissura();
        fissuraSemImagem.setTipo("VERTICAL");

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissuraSemImagem);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        verify(imagemRepository, never()).findById(anyLong());
        verify(fissuraRepository, never()).save(any(Fissura.class));
    }

    @Test
    void testCreateFissura_ImagemIdNull() {
        Fissura fissuraComImagemSemId = new Fissura();
        fissuraComImagemSemId.setImagem(new Imagem());

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissuraComImagemSemId);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        verify(imagemRepository, never()).findById(anyLong());
        verify(fissuraRepository, never()).save(any(Fissura.class));
    }

    @Test
    void testCreateFissura_ImagemNotFound() {
        when(imagemRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissura);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
        verify(imagemRepository).findById(1L);
        verify(fissuraRepository, never()).save(any(Fissura.class));
    }

    @Test
    void testCreateFissura_WithoutDataDeteccao() {
        Fissura fissuraSemData = new Fissura();
        fissuraSemData.setImagem(imagem);
        fissuraSemData.setTipo("HORIZONTAL");
        
        when(imagemRepository.findById(1L)).thenReturn(Optional.of(imagem));
        when(fissuraRepository.save(any(Fissura.class))).thenReturn(fissuraSemData);

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissuraSemData);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        verify(imagemRepository).findById(1L);
        verify(fissuraRepository).save(any(Fissura.class));
    }

    @Test
    void testCreateFissura_Exception() {
        when(imagemRepository.findById(1L)).thenReturn(Optional.of(imagem));
        when(fissuraRepository.save(any(Fissura.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Fissura> response = fissuraController.createFissura(fissura);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para aprovarFissura()
    @Test
    void testAprovarFissura_Success() {
        Map<String, Object> request = new HashMap<>();
        request.put("aprovadoPor", "usuario_teste");

        when(fissuraRepository.findById(1L)).thenReturn(Optional.of(fissura));
        when(fissuraRepository.save(any(Fissura.class))).thenReturn(fissura);

        ResponseEntity<Fissura> response = fissuraController.aprovarFissura(1L, request);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(fissura, response.getBody());
        assertTrue(fissura.getAprovado());
        assertEquals("usuario_teste", fissura.getAprovadoPor());
        verify(fissuraRepository).findById(1L);
        verify(fissuraRepository).save(any(Fissura.class));
    }

    @Test
    void testAprovarFissura_NotFound() {
        Map<String, Object> request = new HashMap<>();
        request.put("aprovadoPor", "usuario_teste");

        when(fissuraRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Fissura> response = fissuraController.aprovarFissura(1L, request);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
        verify(fissuraRepository).findById(1L);
        verify(fissuraRepository, never()).save(any(Fissura.class));
    }

    @Test
    void testAprovarFissura_Exception() {
        Map<String, Object> request = new HashMap<>();
        request.put("aprovadoPor", "usuario_teste");

        when(fissuraRepository.findById(1L)).thenReturn(Optional.of(fissura));
        when(fissuraRepository.save(any(Fissura.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Fissura> response = fissuraController.aprovarFissura(1L, request);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para getPorcentagemPorTipo()
    @Test
    void testGetPorcentagemPorTipo_Success() {
        when(fissuraService.getPorcentagemPorTipo(1)).thenReturn(porcentagemDTO);

        FissuraPorcentagemDTO response = fissuraController.getPorcentagemPorTipo(1);

        assertEquals(porcentagemDTO, response);
        verify(fissuraService).getPorcentagemPorTipo(1);
    }

    // Testes para getFissurasByProjeto()
    @Test
    void testGetFissurasByProjeto_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        List<Fachada> fachadas = Arrays.asList(fachada);
        List<Imagem> imagens = Arrays.asList(imagem);
        List<Fissura> fissuras = Arrays.asList(fissura);

        when(projetoRepository.findById(1)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);
        when(fachadaRepository.findByEdificio(edificio)).thenReturn(fachadas);
        when(imagemRepository.findByFachada(fachada)).thenReturn(imagens);
        when(fissuraRepository.findByImagem(imagem)).thenReturn(fissuras);

        List<Fissura> response = fissuraController.getFissurasByProjeto(1);

        assertEquals(fissuras, response);
        verify(projetoRepository).findById(1);
        verify(edificioRepository).findByProjeto(projeto);
        verify(fachadaRepository).findByEdificio(edificio);
        verify(imagemRepository).findByFachada(fachada);
        verify(fissuraRepository).findByImagem(imagem);
    }

    @Test
    void testGetFissurasByProjeto_ProjectNotFound() {
        when(projetoRepository.findById(1)).thenReturn(Optional.empty());

        try {
            fissuraController.getFissurasByProjeto(1);
        } catch (RuntimeException e) {
            assertEquals("Projeto not found with id: 1", e.getMessage());
        }

        verify(projetoRepository).findById(1);
        verify(edificioRepository, never()).findByProjeto(any());
    }

    // Testes para getFissurasDetalhadasByProjeto()
    @Test
    void testGetFissurasDetalhadasByProjeto_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        List<Fachada> fachadas = Arrays.asList(fachada);
        List<Imagem> imagens = Arrays.asList(imagem);
        List<Fissura> fissuras = Arrays.asList(fissura);

        when(projetoRepository.findById(1)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);
        when(fachadaRepository.findByEdificio(edificio)).thenReturn(fachadas);
        when(imagemRepository.findByFachada(fachada)).thenReturn(imagens);
        when(fissuraRepository.findByImagem(imagem)).thenReturn(fissuras);

        List<FissuraDetalheDTO> response = fissuraController.getFissurasDetalhadasByProjeto(1);

        assertEquals(1, response.size());
        FissuraDetalheDTO dto = response.get(0);
        assertEquals(fissura.getId(), dto.getId());
        assertEquals(fissura.getTipo(), dto.getTipo());
        assertEquals(fissura.getCoordenadas(), dto.getCoordenadas());
        assertEquals(fissura.getGravidade(), dto.getGravidade());
        assertEquals(fissura.getConfianca(), dto.getConfianca());
        assertEquals(imagem.getCaminhoArquivo(), dto.getNomeImagem());
        assertEquals(fissura.getAprovado(), dto.getAprovado());

        verify(projetoRepository).findById(1);
        verify(edificioRepository).findByProjeto(projeto);
        verify(fachadaRepository).findByEdificio(edificio);
        verify(imagemRepository).findByFachada(fachada);
        verify(fissuraRepository).findByImagem(imagem);
    }

    @Test
    void testGetFissurasDetalhadasByProjeto_ProjectNotFound() {
        when(projetoRepository.findById(1)).thenReturn(Optional.empty());

        try {
            fissuraController.getFissurasDetalhadasByProjeto(1);
        } catch (RuntimeException e) {
            assertEquals("Projeto not found with id: 1", e.getMessage());
        }

        verify(projetoRepository).findById(1);
        verify(edificioRepository, never()).findByProjeto(any());
    }

    @Test
    void testGetFissurasDetalhadasByProjeto_WithNullDataDeteccao() {
        fissura.setDataDeteccao(null);
        List<Edificio> edificios = Arrays.asList(edificio);
        List<Fachada> fachadas = Arrays.asList(fachada);
        List<Imagem> imagens = Arrays.asList(imagem);
        List<Fissura> fissuras = Arrays.asList(fissura);

        when(projetoRepository.findById(1)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);
        when(fachadaRepository.findByEdificio(edificio)).thenReturn(fachadas);
        when(imagemRepository.findByFachada(fachada)).thenReturn(imagens);
        when(fissuraRepository.findByImagem(imagem)).thenReturn(fissuras);

        List<FissuraDetalheDTO> response = fissuraController.getFissurasDetalhadasByProjeto(1);

        assertEquals(1, response.size());
        assertNull(response.get(0).getDataDeteccao());
    }

    // Testes para getFissurasDetalhesByImagem()
    @Test
    void testGetFissurasDetalhesByImagem_Success() {
        List<Fissura> fissuras = Arrays.asList(fissura);

        when(imagemRepository.findById(1L)).thenReturn(Optional.of(imagem));
        when(fissuraRepository.findByImagem(imagem)).thenReturn(fissuras);

        ResponseEntity<List<FissuraDetalheDTO>> response = fissuraController.getFissurasDetalhesByImagem(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(1, response.getBody().size());
        
        FissuraDetalheDTO dto = response.getBody().get(0);
        assertEquals(fissura.getId(), dto.getId());
        assertEquals(fissura.getTipo(), dto.getTipo());
        assertEquals(fissura.getCoordenadas(), dto.getCoordenadas());
        assertEquals(fissura.getGravidade(), dto.getGravidade());
        assertEquals(fissura.getConfianca(), dto.getConfianca());
        assertEquals(imagem.getCaminhoArquivo(), dto.getNomeImagem());

        verify(imagemRepository).findById(1L);
        verify(fissuraRepository).findByImagem(imagem);
    }

    @Test
    void testGetFissurasDetalhesByImagem_ImagemNotFound() {
        when(imagemRepository.findById(1L)).thenReturn(Optional.empty());

        try {
            fissuraController.getFissurasDetalhesByImagem(1L);
        } catch (RuntimeException e) {
            assertEquals("Imagem not found with id: 1", e.getMessage());
        }

        verify(imagemRepository).findById(1L);
        verify(fissuraRepository, never()).findByImagem(any());
    }

    @Test
    void testGetFissurasDetalhesByImagem_EmptyFissuras() {
        when(imagemRepository.findById(1L)).thenReturn(Optional.of(imagem));
        when(fissuraRepository.findByImagem(imagem)).thenReturn(new ArrayList<>());

        ResponseEntity<List<FissuraDetalheDTO>> response = fissuraController.getFissurasDetalhesByImagem(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertTrue(response.getBody().isEmpty());

        verify(imagemRepository).findById(1L);
        verify(fissuraRepository).findByImagem(imagem);
    }
}