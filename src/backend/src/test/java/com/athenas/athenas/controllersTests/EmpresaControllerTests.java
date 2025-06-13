package com.athenas.athenas.controllersTests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;

import com.athenas.athenas.controller.EmpresaController;
import com.athenas.athenas.dto.EmpresaDTO;
import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.service.EmpresaService;

@ExtendWith(MockitoExtension.class)
public class EmpresaControllerTests {

    @Mock
    private EmpresaService empresaService;

    @InjectMocks
    private EmpresaController empresaController;

    private Empresa empresa;
    private EmpresaDTO empresaDTO;

    @BeforeEach
    void setUp() {
        empresa = new Empresa();
        empresa.setId(1L);
        empresa.setNome("Empresa Teste");
        empresa.setCnpj("12.345.678/0001-95");

        empresaDTO = new EmpresaDTO();
        empresaDTO.setNome("Empresa Teste");
        empresaDTO.setCnpj("12.345.678/0001-95");
    }

    // Testes para createEmpresa()
    @Test
    void testCreateEmpresa_Success() {
        when(empresaService.createEmpresa(any(EmpresaDTO.class))).thenReturn(empresa);

        Empresa response = empresaController.createEmpresa(empresaDTO);

        assertEquals(empresa, response);
        verify(empresaService).createEmpresa(any(EmpresaDTO.class));
    }

    @Test
    void testCreateEmpresa_Exception() {
        when(empresaService.createEmpresa(any(EmpresaDTO.class)))
            .thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.createEmpresa(empresaDTO);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).createEmpresa(any(EmpresaDTO.class));
    }

    // Testes para getAllEmpresas()
    @Test
    void testGetAllEmpresas_Success() {
        List<Empresa> empresas = Arrays.asList(empresa);
        when(empresaService.getAllEmpresas()).thenReturn(empresas);

        List<Empresa> response = empresaController.getAllEmpresas();

        assertEquals(empresas, response);
        assertEquals(1, response.size());
        verify(empresaService).getAllEmpresas();
    }

    @Test
    void testGetAllEmpresas_EmptyList() {
        List<Empresa> empresasVazias = new ArrayList<>();
        when(empresaService.getAllEmpresas()).thenReturn(empresasVazias);

        List<Empresa> response = empresaController.getAllEmpresas();

        assertEquals(empresasVazias, response);
        assertEquals(0, response.size());
        verify(empresaService).getAllEmpresas();
    }

    @Test
    void testGetAllEmpresas_Exception() {
        when(empresaService.getAllEmpresas()).thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.getAllEmpresas();
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).getAllEmpresas();
    }

    // Testes para getEmpresaById()
    @Test
    void testGetEmpresaById_Success() {
        when(empresaService.getEmpresaById(1)).thenReturn(empresa);

        Empresa response = empresaController.getEmpresaById(1);

        assertEquals(empresa, response);
        verify(empresaService).getEmpresaById(1);
    }

    @Test
    void testGetEmpresaById_NotFound() {
        when(empresaService.getEmpresaById(999)).thenReturn(null);

        Empresa response = empresaController.getEmpresaById(999);

        assertNull(response);
        verify(empresaService).getEmpresaById(999);
    }

    @Test
    void testGetEmpresaById_Exception() {
        when(empresaService.getEmpresaById(anyInt())).thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.getEmpresaById(1);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).getEmpresaById(1);
    }

    // Testes para getEmpresaByNome()
    @Test
    void testGetEmpresaByNome_Success() {
        when(empresaService.getEmpresaByNome("Empresa Teste")).thenReturn(empresa);

        Empresa response = empresaController.getEmpresaByNome("Empresa Teste");

        assertEquals(empresa, response);
        verify(empresaService).getEmpresaByNome("Empresa Teste");
    }

    @Test
    void testGetEmpresaByNome_NotFound() {
        when(empresaService.getEmpresaByNome("Empresa Inexistente")).thenReturn(null);

        Empresa response = empresaController.getEmpresaByNome("Empresa Inexistente");

        assertNull(response);
        verify(empresaService).getEmpresaByNome("Empresa Inexistente");
    }

    @Test
    void testGetEmpresaByNome_Exception() {
        when(empresaService.getEmpresaByNome(anyString())).thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.getEmpresaByNome("Empresa Teste");
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).getEmpresaByNome("Empresa Teste");
    }

    // Testes para getEmpresaByCNPJ()
    @Test
    void testGetEmpresaByCNPJ_Success() {
        when(empresaService.getEmpresaByCNPJ("12.345.678/0001-95")).thenReturn(empresa);

        Empresa response = empresaController.getEmpresaByCNPJ("12.345.678/0001-95");

        assertEquals(empresa, response);
        verify(empresaService).getEmpresaByCNPJ("12.345.678/0001-95");
    }

    @Test
    void testGetEmpresaByCNPJ_NotFound() {
        when(empresaService.getEmpresaByCNPJ("99.999.999/9999-99")).thenReturn(null);

        Empresa response = empresaController.getEmpresaByCNPJ("99.999.999/9999-99");

        assertNull(response);
        verify(empresaService).getEmpresaByCNPJ("99.999.999/9999-99");
    }

    @Test
    void testGetEmpresaByCNPJ_Exception() {
        when(empresaService.getEmpresaByCNPJ(anyString())).thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.getEmpresaByCNPJ("12.345.678/0001-95");
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).getEmpresaByCNPJ("12.345.678/0001-95");
    }

    // Testes para updateEmpresa()
    @Test
    void testUpdateEmpresa_Success() {
        Empresa empresaAtualizada = new Empresa();
        empresaAtualizada.setId(1L);
        empresaAtualizada.setNome("Empresa Atualizada");
        empresaAtualizada.setCnpj("12.345.678/0001-95");

        when(empresaService.updateEmpresa(1L, empresaDTO)).thenReturn(empresaAtualizada);

        Empresa response = empresaController.updateEmpresa(1L, empresaDTO);

        assertEquals(empresaAtualizada, response);
        assertEquals("Empresa Atualizada", response.getNome());
        verify(empresaService).updateEmpresa(1L, empresaDTO);
    }

    @Test
    void testUpdateEmpresa_NotFound() {
        when(empresaService.updateEmpresa(999L, empresaDTO)).thenReturn(null);

        Empresa response = empresaController.updateEmpresa(999L, empresaDTO);

        assertNull(response);
        verify(empresaService).updateEmpresa(999L, empresaDTO);
    }

    @Test
    void testUpdateEmpresa_Exception() {
        when(empresaService.updateEmpresa(anyLong(), any(EmpresaDTO.class)))
            .thenThrow(new RuntimeException("Service error"));

        try {
            empresaController.updateEmpresa(1L, empresaDTO);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).updateEmpresa(1L, empresaDTO);
    }

    // Testes para deleteEmpresa()
    @Test
    void testDeleteEmpresa_Success() {
        empresaController.deleteEmpresa(1L);

        verify(empresaService).deleteEmpresa(1L);
    }

    @Test
    void testDeleteEmpresa_Exception() {
        doThrow(new RuntimeException("Service error")).when(empresaService).deleteEmpresa(1L);

        try {
            empresaController.deleteEmpresa(1L);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(empresaService).deleteEmpresa(1L);
    }

    @Test
    void testDeleteEmpresa_NotFound() {
        doThrow(new RuntimeException("Empresa não encontrada")).when(empresaService).deleteEmpresa(999L);

        try {
            empresaController.deleteEmpresa(999L);
        } catch (RuntimeException e) {
            assertEquals("Empresa não encontrada", e.getMessage());
        }
        
        verify(empresaService).deleteEmpresa(999L);
    }
}