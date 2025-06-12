package com.athenas.athenas.controllersTests;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.athenas.athenas.controller.ViewProjetoController;
import com.athenas.athenas.dto.UpdateViewProjetoRequest;
import com.athenas.athenas.dto.ViewProjetoRequestDTO;
import com.athenas.athenas.dto.ViewProjetoResponseDTO;
import com.athenas.athenas.service.ViewProjetoService;

@ExtendWith(MockitoExtension.class)
public class ViewProjetoControllerTests {

    @Mock
    private ViewProjetoService viewProjetoService;

    @InjectMocks
    private ViewProjetoController viewProjetoController;

    private ViewProjetoRequestDTO viewProjetoRequestDTO;
    private ViewProjetoResponseDTO viewProjetoResponseDTO;
    private UpdateViewProjetoRequest updateViewProjetoRequest;

    @BeforeEach
    void setUp() {
        viewProjetoRequestDTO = new ViewProjetoRequestDTO();
        viewProjetoRequestDTO.setIdProjeto(1);

        viewProjetoResponseDTO = createMockViewProjetoResponseDTO(1, "Projeto Teste", "Descrição do projeto teste");

        updateViewProjetoRequest = new UpdateViewProjetoRequest();
        updateViewProjetoRequest.setIdProjeto(1);
        updateViewProjetoRequest.setViewProjetoResponseDTO(viewProjetoResponseDTO);
    }

    private ViewProjetoResponseDTO createMockViewProjetoResponseDTO(int id, String nome, String descricao) {
        ViewProjetoResponseDTO dto = new ViewProjetoResponseDTO();
        dto.setNome(nome);
        dto.setDescricao(descricao);
        return dto;
    }

    // Testes para viewProjeto()
    @Test
    void testViewProjeto_Success() {
        when(viewProjetoService.ReadProjectData(1)).thenReturn(viewProjetoResponseDTO);

        ResponseEntity<ViewProjetoResponseDTO> response = viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(viewProjetoResponseDTO, response.getBody());
        assertEquals("Projeto Teste", response.getBody().getNome());
        verify(viewProjetoService).ReadProjectData(1);
    }

    @Test
    void testViewProjeto_ServiceException() {
        when(viewProjetoService.ReadProjectData(anyInt()))
            .thenThrow(new RuntimeException("Service error"));

        ResponseEntity<ViewProjetoResponseDTO> response = viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        verify(viewProjetoService).ReadProjectData(1);
    }

    @Test
    void testViewProjeto_ServiceReturnsNull() {
        when(viewProjetoService.ReadProjectData(1)).thenReturn(null);

        ResponseEntity<ViewProjetoResponseDTO> response = viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNull(response.getBody());
        verify(viewProjetoService).ReadProjectData(1);
    }

    @Test
    void testViewProjeto_InvalidProjectId() {
        ViewProjetoRequestDTO invalidRequest = new ViewProjetoRequestDTO();
        invalidRequest.setIdProjeto(999);
        
        when(viewProjetoService.ReadProjectData(anyInt()))
            .thenThrow(new IllegalArgumentException("Projeto não encontrado"));
        
        ResponseEntity<ViewProjetoResponseDTO> response = viewProjetoController.viewProjeto(invalidRequest);
        
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        
        verify(viewProjetoService).ReadProjectData(anyInt());
    }

    @Test
    void testViewProjeto_NullPointerException() {
        ViewProjetoRequestDTO nullRequest = new ViewProjetoRequestDTO();
        nullRequest.setIdProjeto(null);
        
        ResponseEntity<ViewProjetoResponseDTO> response = viewProjetoController.viewProjeto(nullRequest);
        
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        
        verifyNoInteractions(viewProjetoService);
    }

    // Testes para updateViewProjeto()
    @Test
    void testUpdateViewProjeto_Success() {
        ViewProjetoResponseDTO updatedResponse = createMockViewProjetoResponseDTO(1, "Projeto Atualizado", "Descrição atualizada");

        when(viewProjetoService.UpdateProjectData(1, viewProjetoResponseDTO))
            .thenReturn(updatedResponse);

        ViewProjetoResponseDTO response = viewProjetoController.updateViewProjeto(updateViewProjetoRequest);

        assertEquals(updatedResponse, response);
        assertEquals("Projeto Atualizado", response.getNome());
        assertEquals("Descrição atualizada", response.getDescricao());
        verify(viewProjetoService).UpdateProjectData(1, viewProjetoResponseDTO);
    }

    @Test
    void testUpdateViewProjeto_ServiceReturnsNull() {
        when(viewProjetoService.UpdateProjectData(1, viewProjetoResponseDTO))
            .thenReturn(null);

        ViewProjetoResponseDTO response = viewProjetoController.updateViewProjeto(updateViewProjetoRequest);

        assertNull(response);
        verify(viewProjetoService).UpdateProjectData(1, viewProjetoResponseDTO);
    }

    @Test
    void testUpdateViewProjeto_ProjectNotFound() {
        UpdateViewProjetoRequest invalidRequest = new UpdateViewProjetoRequest();
        invalidRequest.setIdProjeto(999);
        invalidRequest.setViewProjetoResponseDTO(viewProjetoResponseDTO);

        when(viewProjetoService.UpdateProjectData(999, viewProjetoResponseDTO))
            .thenThrow(new RuntimeException("Projeto não encontrado"));

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            viewProjetoController.updateViewProjeto(invalidRequest);
        });

        assertEquals("Projeto não encontrado", exception.getMessage());
        verify(viewProjetoService).UpdateProjectData(999, viewProjetoResponseDTO);
    }

    @Test
    void testUpdateViewProjeto_NullRequest() {
        UpdateViewProjetoRequest nullRequest = new UpdateViewProjetoRequest();
        nullRequest.setIdProjeto(1); // Fixed: Use concrete value instead of any()
        nullRequest.setViewProjetoResponseDTO(null);

        when(viewProjetoService.UpdateProjectData(1, null))
            .thenThrow(new IllegalArgumentException("Parâmetros inválidos"));

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            viewProjetoController.updateViewProjeto(nullRequest);
        });

        assertEquals("Parâmetros inválidos", exception.getMessage());
        verify(viewProjetoService).UpdateProjectData(1, null);
    }

    @Test
    void testUpdateViewProjeto_ServiceException() {
        when(viewProjetoService.UpdateProjectData(anyInt(), any(ViewProjetoResponseDTO.class)))
            .thenThrow(new RuntimeException("Erro interno do serviço"));

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            viewProjetoController.updateViewProjeto(updateViewProjetoRequest);
        });

        assertEquals("Erro interno do serviço", exception.getMessage());
        verify(viewProjetoService).UpdateProjectData(1, viewProjetoResponseDTO);
    }

    @Test
    void testUpdateViewProjeto_PartialUpdate() {
        ViewProjetoResponseDTO partialResponse = createMockViewProjetoResponseDTO(1, "Projeto Parcialmente Atualizado", null);

        UpdateViewProjetoRequest partialRequest = new UpdateViewProjetoRequest();
        partialRequest.setIdProjeto(1);
        partialRequest.setViewProjetoResponseDTO(partialResponse);

        when(viewProjetoService.UpdateProjectData(1, partialResponse))
            .thenReturn(partialResponse);

        ViewProjetoResponseDTO response = viewProjetoController.updateViewProjeto(partialRequest);

        assertEquals(partialResponse, response);
        assertEquals("Projeto Parcialmente Atualizado", response.getNome());
        assertNull(response.getDescricao());
        verify(viewProjetoService).UpdateProjectData(1, partialResponse);
    }
}