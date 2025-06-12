package com.athenas.athenas.controllersTests;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
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
        viewProjetoRequestDTO.setIdProjeto(1L);

        viewProjetoResponseDTO = new ViewProjetoResponseDTO();
        viewProjetoResponseDTO.setId(1L);
        viewProjetoResponseDTO.setNome("Projeto Teste");

        updateViewProjetoRequest = new UpdateViewProjetoRequest();
        updateViewProjetoRequest.setIdProjeto(1L);
        updateViewProjetoRequest.setViewProjetoResponseDTO(viewProjetoResponseDTO);
    }

    // Testes para viewProjeto()
    @Test
    void testViewProjeto_Success() {
        when(viewProjetoService.ReadProjectData(1L)).thenReturn(viewProjetoResponseDTO);

        ResponseEntity<ViewProjetoResponseDTO> response = 
            viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(viewProjetoResponseDTO, response.getBody());
        verify(viewProjetoService).ReadProjectData(1L);
    }

    @Test
    void testViewProjeto_ServiceException() {
        when(viewProjetoService.ReadProjectData(1L)).thenThrow(new RuntimeException("Service error"));

        ResponseEntity<ViewProjetoResponseDTO> response = 
            viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        verify(viewProjetoService).ReadProjectData(1L);
    }

    @Test
    void testViewProjeto_WithNullId() {
        viewProjetoRequestDTO.setIdProjeto(null);

        ResponseEntity<ViewProjetoResponseDTO> response = 
            viewProjetoController.viewProjeto(viewProjetoRequestDTO);

        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
        verify(viewProjetoService, never()).ReadProjectData(anyLong());
    }

    // Testes para updateViewProjeto()
    @Test
    void testUpdateViewProjeto_Success() {
        when(viewProjetoService.UpdateProjectData(1L, viewProjetoResponseDTO))
            .thenReturn(viewProjetoResponseDTO);

        ViewProjetoResponseDTO response = 
            viewProjetoController.updateViewProjeto(updateViewProjetoRequest);

        assertEquals(viewProjetoResponseDTO, response);
        verify(viewProjetoService).UpdateProjectData(1L, viewProjetoResponseDTO);
    }

    @Test
    void testUpdateViewProjeto_WithNullId() {
        updateViewProjetoRequest.setIdProjeto(null);

        ViewProjetoResponseDTO response = 
            viewProjetoController.updateViewProjeto(updateViewProjetoRequest);

        assertNull(response);
        verify(viewProjetoService, never()).UpdateProjectData(anyLong(), any());
    }

    @Test
    void testUpdateViewProjeto_WithNullViewProjeto() {
        updateViewProjetoRequest.setViewProjetoResponseDTO(null);

        ViewProjetoResponseDTO response = 
            viewProjetoController.updateViewProjeto(updateViewProjetoRequest);

        assertNull(response);
        verify(viewProjetoService, never()).UpdateProjectData(anyLong(), any());
    }

    @Test
    void testUpdateViewProjeto_ServiceException() {
        when(viewProjetoService.UpdateProjectData(1L, viewProjetoResponseDTO))
            .thenThrow(new RuntimeException("Service error"));

        try {
            viewProjetoController.updateViewProjeto(updateViewProjetoRequest);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(viewProjetoService).UpdateProjectData((int) 1L, viewProjetoResponseDTO);
    }
}