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
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.athenas.athenas.controller.ProjetoController;
import com.athenas.athenas.dto.ProjetoDTO;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.service.ProjetoService;

@ExtendWith(MockitoExtension.class)
public class ProjetoControllerTests {

    @Mock
    private ProjetoService projetoService;

    @InjectMocks
    private ProjetoController projetoController;

    private Projeto projeto;
    private ProjetoDTO projetoDTO;

    @BeforeEach
    void setUp() {
        projeto = new Projeto();
        projeto.setId(1L);
        projeto.setNome("Projeto Teste");
        projeto.setDescricao("Descrição do projeto teste");
        projeto.setStatus("ATIVO");

        projetoDTO = new ProjetoDTO();
        projetoDTO.setNome("Projeto DTO Teste");
        projetoDTO.setDescricao("Descrição do projeto DTO teste");
        projetoDTO.setStatus("ATIVO");
        projetoDTO.setEmpresa(1L);
    }

    // Testes para listAllProjects()
    @Test
    void testListAllProjects_WithoutNome() {
        List<Projeto> projetos = Arrays.asList(projeto);
        when(projetoService.findAll()).thenReturn(projetos);

        List<Projeto> response = projetoController.listAllProjects(null);

        assertEquals(projetos, response);
        verify(projetoService).findAll();
        verify(projetoService, never()).buscarPorNomeSemAcento(anyString());
    }

    @Test
    void testListAllProjects_WithEmptyNome() {
        List<Projeto> projetos = Arrays.asList(projeto);
        when(projetoService.findAll()).thenReturn(projetos);

        List<Projeto> response = projetoController.listAllProjects("");

        assertEquals(projetos, response);
        verify(projetoService).findAll();
        verify(projetoService, never()).buscarPorNomeSemAcento(anyString());
    }

    @Test
    void testListAllProjects_WithNome() {
        List<Projeto> projetos = Arrays.asList(projeto);
        when(projetoService.buscarPorNomeSemAcento("Projeto Teste")).thenReturn(projetos);

        List<Projeto> response = projetoController.listAllProjects("Projeto Teste");

        assertEquals(projetos, response);
        verify(projetoService).buscarPorNomeSemAcento("Projeto Teste");
        verify(projetoService, never()).findAll();
    }

    @Test
    void testListAllProjects_WithNome_EmptyResult() {
        List<Projeto> projetosVazios = new ArrayList<>();
        when(projetoService.buscarPorNomeSemAcento("Projeto Inexistente")).thenReturn(projetosVazios);

        List<Projeto> response = projetoController.listAllProjects("Projeto Inexistente");

        assertEquals(projetosVazios, response);
        verify(projetoService).buscarPorNomeSemAcento("Projeto Inexistente");
    }

    // Testes para createProject()
    @Test
    void testCreateProject_Success() {
        when(projetoService.saveWithEmpresaId(any(Projeto.class), anyLong())).thenReturn(projeto);

        Projeto response = projetoController.createProject(projetoDTO);

        assertEquals(projeto, response);
        verify(projetoService).saveWithEmpresaId(any(Projeto.class), anyLong());
    }

    @Test
    void testCreateProject_WithNullValues() {
        ProjetoDTO projetoDTOComNulos = new ProjetoDTO();
        projetoDTOComNulos.setNome(null);
        projetoDTOComNulos.setDescricao(null);
        projetoDTOComNulos.setStatus(null);
        projetoDTOComNulos.setEmpresa(1L);

        Projeto projetoComNulos = new Projeto();
        projetoComNulos.setId(1L);
        projetoComNulos.setNome(null);
        projetoComNulos.setDescricao(null);
        projetoComNulos.setStatus(null);

        when(projetoService.saveWithEmpresaId(any(Projeto.class), anyLong())).thenReturn(projetoComNulos);

        Projeto response = projetoController.createProject(projetoDTOComNulos);

        assertEquals(projetoComNulos, response);
        verify(projetoService).saveWithEmpresaId(any(Projeto.class), anyLong());
    }

    // Testes para getProjectById()
    @Test
    void testGetProjectById_Success() {
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));

        ResponseEntity<Projeto> response = projetoController.getProjectById(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(projeto, response.getBody());
        verify(projetoService).findById(1L);
    }

    @Test
    void testGetProjectById_NotFound() {
        when(projetoService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Projeto> response = projetoController.getProjectById(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoService).findById(1L);
    }

    // Testes para updateProject()
    @Test
    void testUpdateProject_Success() {
        Projeto projetoAtualizado = new Projeto();
        projetoAtualizado.setId(1L);
        projetoAtualizado.setNome(projetoDTO.getNome());
        projetoAtualizado.setDescricao(projetoDTO.getDescricao());
        projetoAtualizado.setStatus(projetoDTO.getStatus());

        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        when(projetoService.updateWithEmpresaId(any(Projeto.class), anyLong())).thenReturn(projetoAtualizado);

        ResponseEntity<Projeto> response = projetoController.updateProject(1L, projetoDTO);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(projetoAtualizado, response.getBody());
        verify(projetoService, times(2)).findById(1L); // Chamado 2 vezes no controller
        verify(projetoService).updateWithEmpresaId(any(Projeto.class), anyLong());
    }

    @Test
    void testUpdateProject_NotFound() {
        when(projetoService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Projeto> response = projetoController.updateProject(1L, projetoDTO);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoService).findById(1L);
        verify(projetoService, never()).updateWithEmpresaId(any(Projeto.class), anyLong());
    }

    @Test
    void testUpdateProject_WithNullValues() {
        ProjetoDTO projetoDTOComNulos = new ProjetoDTO();
        projetoDTOComNulos.setNome(null);
        projetoDTOComNulos.setDescricao(null);
        projetoDTOComNulos.setStatus(null);
        projetoDTOComNulos.setEmpresa(1L);

        Projeto projetoAtualizado = new Projeto();
        projetoAtualizado.setId(1L);
        projetoAtualizado.setNome(null);
        projetoAtualizado.setDescricao(null);
        projetoAtualizado.setStatus(null);

        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        when(projetoService.updateWithEmpresaId(any(Projeto.class), anyLong())).thenReturn(projetoAtualizado);

        ResponseEntity<Projeto> response = projetoController.updateProject(1L, projetoDTOComNulos);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(projetoAtualizado, response.getBody());
        verify(projetoService, times(2)).findById(1L); // Chamado 2 vezes no controller
        verify(projetoService).updateWithEmpresaId(any(Projeto.class), anyLong());
    }

    // Testes adicionais para cenários de exceção (se necessário)
    @Test
    void testCreateProject_ServiceException() {
        when(projetoService.saveWithEmpresaId(any(Projeto.class), anyLong()))
                .thenThrow(new RuntimeException("Service error"));

        try {
            projetoController.createProject(projetoDTO);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(projetoService).saveWithEmpresaId(any(Projeto.class), anyLong());
    }

    @Test
    void testGetProjectById_ServiceException() {
        when(projetoService.findById(1L)).thenThrow(new RuntimeException("Service error"));

        try {
            projetoController.getProjectById(1L);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(projetoService).findById(1L);
    }

    @Test
    void testUpdateProject_ServiceException() {
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        when(projetoService.updateWithEmpresaId(any(Projeto.class), anyLong()))
                .thenThrow(new RuntimeException("Service error"));

        try {
            projetoController.updateProject(1L, projetoDTO);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(projetoService, times(2)).findById(1L); // Chamado 2 vezes no controller
        verify(projetoService).updateWithEmpresaId(any(Projeto.class), anyLong());
    }
}