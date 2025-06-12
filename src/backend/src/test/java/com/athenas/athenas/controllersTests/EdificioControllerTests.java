package com.athenas.athenas.controllersTests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.athenas.athenas.controller.EdificioController;
import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.ProjetoRepository;

@ExtendWith(MockitoExtension.class)
public class EdificioControllerTests {

    @Mock
    private EdificioRepository edificioRepository;

    @Mock
    private ProjetoRepository projetoRepository;

    @InjectMocks
    private EdificioController edificioController;

    private Edificio edificio;
    private Projeto projeto;
    private Fachada fachada;

    @BeforeEach
    void setUp() {
        projeto = new Projeto();
        projeto.setId(1L);
        projeto.setNome("Projeto Teste");

        fachada = new Fachada();
        fachada.setId(1L);
        fachada.setNome("Fachada Norte");
        fachada.setDescricao("Descrição da fachada norte");

        edificio = new Edificio();
        edificio.setId(1L);
        edificio.setNome("Edificio Teste");
        edificio.setProjeto(projeto);
        edificio.setFachadas(Arrays.asList(fachada));
    }

    // Testes para createEdificio()
    @Test
    void testCreateEdificio_Success() {
        when(edificioRepository.save(any(Edificio.class))).thenReturn(edificio);

        ResponseEntity<Edificio> response = edificioController.createEdificio(edificio);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals(edificio, response.getBody());
        verify(edificioRepository).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificio_WithFachadaEmptyName() {
        Fachada fachadaSemNome = new Fachada();
        fachadaSemNome.setDescricao("Descrição teste");
        fachadaSemNome.setNome("");
        
        Edificio edificioComFachadaSemNome = new Edificio();
        edificioComFachadaSemNome.setFachadas(Arrays.asList(fachadaSemNome));

        when(edificioRepository.save(any(Edificio.class))).thenReturn(edificioComFachadaSemNome);

        ResponseEntity<Edificio> response = edificioController.createEdificio(edificioComFachadaSemNome);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals("Descrição teste", fachadaSemNome.getNome());
        verify(edificioRepository).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificio_Exception() {
        when(edificioRepository.save(any(Edificio.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Edificio> response = edificioController.createEdificio(edificio);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para createEdificioForProject()
    @Test
    void testCreateEdificioForProject_Success() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.save(any(Edificio.class))).thenReturn(edificio);

        ResponseEntity<Edificio> response = edificioController.createEdificioForProject(1L, edificio);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals(edificio, response.getBody());
        assertEquals(projeto, edificio.getProjeto());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificioForProject_ProjectNotFound() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Edificio> response = edificioController.createEdificioForProject(1L, edificio);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository, never()).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificioForProject_Exception() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.save(any(Edificio.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Edificio> response = edificioController.createEdificioForProject(1L, edificio);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para createEdificioForProjectByName()
    @Test
    void testCreateEdificioForProjectByName_Success() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Teste")).thenReturn(Optional.of(projeto));
        when(edificioRepository.save(any(Edificio.class))).thenReturn(edificio);

        ResponseEntity<Edificio> response = edificioController.createEdificioForProjectByName("Projeto Teste", edificio);

        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertEquals(edificio, response.getBody());
        assertEquals(projeto, edificio.getProjeto());
        verify(projetoRepository).findByNomeIgnoreCase("Projeto Teste");
        verify(edificioRepository).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificioForProjectByName_ProjectNotFound() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Inexistente")).thenReturn(Optional.empty());

        ResponseEntity<Edificio> response = edificioController.createEdificioForProjectByName("Projeto Inexistente", edificio);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoRepository).findByNomeIgnoreCase("Projeto Inexistente");
        verify(edificioRepository, never()).save(any(Edificio.class));
    }

    @Test
    void testCreateEdificioForProjectByName_Exception() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Teste")).thenReturn(Optional.of(projeto));
        when(edificioRepository.save(any(Edificio.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Edificio> response = edificioController.createEdificioForProjectByName("Projeto Teste", edificio);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para getAllEdificios()
    @Test
    void testGetAllEdificios_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        when(edificioRepository.findAll()).thenReturn(edificios);

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificios();

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(edificios, response.getBody());
        verify(edificioRepository).findAll();
    }

    @Test
    void testGetAllEdificios_NoContent() {
        when(edificioRepository.findAll()).thenReturn(new ArrayList<>());

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificios();

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        assertNull(response.getBody());
        verify(edificioRepository).findAll();
    }

    @Test
    void testGetAllEdificios_Exception() {
        when(edificioRepository.findAll()).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificios();

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para getEdificioById()
    @Test
    void testGetEdificioById_Success() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.of(edificio));

        ResponseEntity<Edificio> response = edificioController.getEdificioById(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(edificio, response.getBody());
        verify(edificioRepository).findById(1L);
    }

    @Test
    void testGetEdificioById_NotFound() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Edificio> response = edificioController.getEdificioById(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(edificioRepository).findById(1L);
    }

    @Test
    void testGetEdificioById_Exception() {
        when(edificioRepository.findById(1L)).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Edificio> response = edificioController.getEdificioById(1L);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para getAllEdificiosByProjectName()
    @Test
    void testGetAllEdificiosByProjectName_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        when(projetoRepository.findByNomeIgnoreCase("Projeto Teste")).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProjectName("Projeto Teste");

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(edificios, response.getBody());
        verify(projetoRepository).findByNomeIgnoreCase("Projeto Teste");
        verify(edificioRepository).findByProjeto(projeto);
    }

    @Test
    void testGetAllEdificiosByProjectName_ProjectNotFound() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Inexistente")).thenReturn(Optional.empty());

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProjectName("Projeto Inexistente");

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoRepository).findByNomeIgnoreCase("Projeto Inexistente");
        verify(edificioRepository, never()).findByProjeto(any());
    }

    @Test
    void testGetAllEdificiosByProjectName_NoContent() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Teste")).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(new ArrayList<>());

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProjectName("Projeto Teste");

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        assertNull(response.getBody());
    }

    @Test
    void testGetAllEdificiosByProjectName_Exception() {
        when(projetoRepository.findByNomeIgnoreCase("Projeto Teste")).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProjectName("Projeto Teste");

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para getAllEdificiosByProject()
    @Test
    void testGetAllEdificiosByProject_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProject(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(edificios, response.getBody());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository).findByProjeto(projeto);
    }

    @Test
    void testGetAllEdificiosByProject_ProjectNotFound() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProject(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository, never()).findByProjeto(any());
    }

    @Test
    void testGetAllEdificiosByProject_NoContent() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(new ArrayList<>());

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProject(1L);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        assertNull(response.getBody());
    }

    @Test
    void testGetAllEdificiosByProject_Exception() {
        when(projetoRepository.findById(1L)).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<List<Edificio>> response = edificioController.getAllEdificiosByProject(1L);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para updateEdificio()
    @Test
    void testUpdateEdificio_Success() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.of(edificio));
        when(edificioRepository.save(any(Edificio.class))).thenReturn(edificio);

        ResponseEntity<Edificio> response = edificioController.updateEdificio(1L, edificio);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(edificio, response.getBody());
        assertEquals(1L, edificio.getId());
        verify(edificioRepository).findById(1L);
        verify(edificioRepository).save(any(Edificio.class));
    }

    @Test
    void testUpdateEdificio_NotFound() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Edificio> response = edificioController.updateEdificio(1L, edificio);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(edificioRepository).findById(1L);
        verify(edificioRepository, never()).save(any(Edificio.class));
    }

    @Test
    void testUpdateEdificio_Exception() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.of(edificio));
        when(edificioRepository.save(any(Edificio.class))).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<Edificio> response = edificioController.updateEdificio(1L, edificio);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    // Testes para deleteEdificio()
    @Test
    void testDeleteEdificio_Success() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.of(edificio));

        ResponseEntity<HttpStatus> response = edificioController.deleteEdificio(1L);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        verify(edificioRepository).findById(1L);
        verify(edificioRepository).deleteById(1L);
    }

    @Test
    void testDeleteEdificio_NotFound() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<HttpStatus> response = edificioController.deleteEdificio(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        verify(edificioRepository).findById(1L);
        verify(edificioRepository, never()).deleteById(any());
    }

    @Test
    void testDeleteEdificio_Exception() {
        when(edificioRepository.findById(1L)).thenReturn(Optional.of(edificio));
        doThrow(new RuntimeException("Database error")).when(edificioRepository).deleteById(1L);

        ResponseEntity<HttpStatus> response = edificioController.deleteEdificio(1L);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
    }

    // Testes para deleteAllEdificiosByProject()
    @Test
    void testDeleteAllEdificiosByProject_Success() {
        List<Edificio> edificios = Arrays.asList(edificio);
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenReturn(edificios);

        ResponseEntity<HttpStatus> response = edificioController.deleteAllEdificiosByProject(1L);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository).findByProjeto(projeto);
        verify(edificioRepository).deleteAll(edificios);
    }

    @Test
    void testDeleteAllEdificiosByProject_ProjectNotFound() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<HttpStatus> response = edificioController.deleteAllEdificiosByProject(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        verify(projetoRepository).findById(1L);
        verify(edificioRepository, never()).findByProjeto(any());
        verify(edificioRepository, never()).deleteAll(any());
    }

    @Test
    void testDeleteAllEdificiosByProject_Exception() {
        when(projetoRepository.findById(1L)).thenReturn(Optional.of(projeto));
        when(edificioRepository.findByProjeto(projeto)).thenThrow(new RuntimeException("Database error"));

        ResponseEntity<HttpStatus> response = edificioController.deleteAllEdificiosByProject(1L);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
    }
}