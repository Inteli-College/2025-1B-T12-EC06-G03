package com.athenas.athenas.controllersTests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.doNothing;
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
        empresa.setCnpj("12.345.678/0001-90");
        empresa.setEmail("contato@empresateste.com");

        empresaDTO = new EmpresaDTO();
        empresaDTO.setNome("Empresa Teste");
        empresaDTO.setCnpj("12.345.678/0001-90");
        empresaDTO.setEmail("contato@empresateste.com");
    }

    // Testes para createEmpresa()
    @Test
    void testCreateEmpresa_Success() {
        when(empresaService.createEmpresa(any(EmpresaDTO.class))).thenReturn(empresa);

        Empresa result = empresaController.createEmpresa(empresaDTO);

        assertNotNull(result);
        assertEquals(empresa.getId(), result.getId());
        assertEquals(empresa.getNome(), result.getNome());
        assertEquals(empresa.getCnpj(), result.getCnpj());
        assertEquals(empresa.getEmail(), result.getEmail());
        verify(empresaService).createEmpresa(empresaDTO);
    }

    @Test
    void testCreateEmpresa_WithNullDTO() {
        when(empresaService.createEmpresa(null)).thenThrow(new IllegalArgumentException("EmpresaDTO cannot be null"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.createEmpresa(null);
        });

        verify(empresaService).createEmpresa(null);
    }

    @Test
    void testCreateEmpresa_ServiceException() {
        when(empresaService.createEmpresa(any(EmpresaDTO.class)))
            .thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.createEmpresa(empresaDTO);
        });

        verify(empresaService).createEmpresa(empresaDTO);
    }

    // Testes para getAllEmpresas()
    @Test
    void testGetAllEmpresas_Success() {
        List<Empresa> empresas = Arrays.asList(empresa);
        when(empresaService.getAllEmpresas()).thenReturn(empresas);

        List<Empresa> result = empresaController.getAllEmpresas();

        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals(empresa, result.get(0));
        verify(empresaService).getAllEmpresas();
    }

    @Test
    void testGetAllEmpresas_EmptyList() {
        List<Empresa> empresasVazias = new ArrayList<>();
        when(empresaService.getAllEmpresas()).thenReturn(empresasVazias);

        List<Empresa> result = empresaController.getAllEmpresas();

        assertNotNull(result);
        assertTrue(result.isEmpty());
        verify(empresaService).getAllEmpresas();
    }

    @Test
    void testGetAllEmpresas_ServiceException() {
        when(empresaService.getAllEmpresas()).thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.getAllEmpresas();
        });

        verify(empresaService).getAllEmpresas();
    }

    // Testes para getEmpresaById()
    @Test
    void testGetEmpresaById_Success() {
        when(empresaService.getEmpresaById(1)).thenReturn(empresa);

        Empresa result = empresaController.getEmpresaById(1);

        assertNotNull(result);
        assertEquals(empresa.getId(), result.getId());
        assertEquals(empresa.getNome(), result.getNome());
        verify(empresaService).getEmpresaById(1);
    }

    @Test
    void testGetEmpresaById_NotFound() {
        when(empresaService.getEmpresaById(999)).thenReturn(null);

        Empresa result = empresaController.getEmpresaById(999);

        assertNull(result);
        verify(empresaService).getEmpresaById(999);
    }

    @Test
    void testGetEmpresaById_ServiceException() {
        when(empresaService.getEmpresaById(1)).thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.getEmpresaById(1);
        });

        verify(empresaService).getEmpresaById(1);
    }

    @Test
    void testGetEmpresaById_NegativeId() {
        when(empresaService.getEmpresaById(-1)).thenThrow(new IllegalArgumentException("ID must be positive"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaById(-1);
        });

        verify(empresaService).getEmpresaById(-1);
    }

    // Testes para getEmpresaByNome()
    @Test
    void testGetEmpresaByNome_Success() {
        when(empresaService.getEmpresaByNome("Empresa Teste")).thenReturn(empresa);

        Empresa result = empresaController.getEmpresaByNome("Empresa Teste");

        assertNotNull(result);
        assertEquals(empresa.getNome(), result.getNome());
        verify(empresaService).getEmpresaByNome("Empresa Teste");
    }

    @Test
    void testGetEmpresaByNome_NotFound() {
        when(empresaService.getEmpresaByNome("Empresa Inexistente")).thenReturn(null);

        Empresa result = empresaController.getEmpresaByNome("Empresa Inexistente");

        assertNull(result);
        verify(empresaService).getEmpresaByNome("Empresa Inexistente");
    }

    @Test
    void testGetEmpresaByNome_EmptyName() {
        when(empresaService.getEmpresaByNome("")).thenThrow(new IllegalArgumentException("Name cannot be empty"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaByNome("");
        });

        verify(empresaService).getEmpresaByNome("");
    }

    @Test
    void testGetEmpresaByNome_NullName() {
        when(empresaService.getEmpresaByNome(null)).thenThrow(new IllegalArgumentException("Name cannot be null"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaByNome(null);
        });

        verify(empresaService).getEmpresaByNome(null);
    }

    @Test
    void testGetEmpresaByNome_ServiceException() {
        when(empresaService.getEmpresaByNome("Empresa Teste")).thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.getEmpresaByNome("Empresa Teste");
        });

        verify(empresaService).getEmpresaByNome("Empresa Teste");
    }

    // Testes para getEmpresaByCNPJ()
    @Test
    void testGetEmpresaByCNPJ_Success() {
        when(empresaService.getEmpresaByCNPJ("12.345.678/0001-90")).thenReturn(empresa);

        Empresa result = empresaController.getEmpresaByCNPJ("12.345.678/0001-90");

        assertNotNull(result);
        assertEquals(empresa.getCnpj(), result.getCnpj());
        verify(empresaService).getEmpresaByCNPJ("12.345.678/0001-90");
    }

    @Test
    void testGetEmpresaByCNPJ_NotFound() {
        when(empresaService.getEmpresaByCNPJ("99.999.999/0001-99")).thenReturn(null);

        Empresa result = empresaController.getEmpresaByCNPJ("99.999.999/0001-99");

        assertNull(result);
        verify(empresaService).getEmpresaByCNPJ("99.999.999/0001-99");
    }

    @Test
    void testGetEmpresaByCNPJ_InvalidFormat() {
        when(empresaService.getEmpresaByCNPJ("123456789")).thenThrow(new IllegalArgumentException("Invalid CNPJ format"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaByCNPJ("123456789");
        });

        verify(empresaService).getEmpresaByCNPJ("123456789");
    }

    @Test
    void testGetEmpresaByCNPJ_EmptyCNPJ() {
        when(empresaService.getEmpresaByCNPJ("")).thenThrow(new IllegalArgumentException("CNPJ cannot be empty"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaByCNPJ("");
        });

        verify(empresaService).getEmpresaByCNPJ("");
    }

    @Test
    void testGetEmpresaByCNPJ_NullCNPJ() {
        when(empresaService.getEmpresaByCNPJ(null)).thenThrow(new IllegalArgumentException("CNPJ cannot be null"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.getEmpresaByCNPJ(null);
        });

        verify(empresaService).getEmpresaByCNPJ(null);
    }

    @Test
    void testGetEmpresaByCNPJ_ServiceException() {
        when(empresaService.getEmpresaByCNPJ("12.345.678/0001-90"))
            .thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.getEmpresaByCNPJ("12.345.678/0001-90");
        });

        verify(empresaService).getEmpresaByCNPJ("12.345.678/0001-90");
    }

    // Testes para updateEmpresa()
    @Test
    void testUpdateEmpresa_Success() {
        Empresa empresaAtualizada = new Empresa();
        empresaAtualizada.setId(1L);
        empresaAtualizada.setNome("Empresa Atualizada");
        empresaAtualizada.setCnpj("12.345.678/0001-90");

        when(empresaService.updateEmpresa(1L, empresaDTO)).thenReturn(empresaAtualizada);

        Empresa result = empresaController.updateEmpresa(1L, empresaDTO);

        assertNotNull(result);
        assertEquals(empresaAtualizada.getId(), result.getId());
        assertEquals("Empresa Atualizada", result.getNome());
        verify(empresaService).updateEmpresa(1L, empresaDTO);
    }

    @Test
    void testUpdateEmpresa_NotFound() {
        when(empresaService.updateEmpresa(999L, empresaDTO)).thenReturn(null);

        Empresa result = empresaController.updateEmpresa(999L, empresaDTO);

        assertNull(result);
        verify(empresaService).updateEmpresa(999L, empresaDTO);
    }

    @Test
    void testUpdateEmpresa_WithNullDTO() {
        when(empresaService.updateEmpresa(1L, null))
            .thenThrow(new IllegalArgumentException("EmpresaDTO cannot be null"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.updateEmpresa(1L, null);
        });

        verify(empresaService).updateEmpresa(1L, null);
    }

    @Test
    void testUpdateEmpresa_NegativeId() {
        when(empresaService.updateEmpresa(-1L, empresaDTO))
            .thenThrow(new IllegalArgumentException("ID must be positive"));

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.updateEmpresa(-1L, empresaDTO);
        });

        verify(empresaService).updateEmpresa(-1L, empresaDTO);
    }

    @Test
    void testUpdateEmpresa_ServiceException() {
        when(empresaService.updateEmpresa(1L, empresaDTO))
            .thenThrow(new RuntimeException("Service error"));

        assertThrows(RuntimeException.class, () -> {
            empresaController.updateEmpresa(1L, empresaDTO);
        });

        verify(empresaService).updateEmpresa(1L, empresaDTO);
    }

    // Testes para deleteEmpresa()
    @Test
    void testDeleteEmpresa_Success() {
        doNothing().when(empresaService).deleteEmpresa(1L);

        assertDoesNotThrow(() -> {
            empresaController.deleteEmpresa(1L);
        });

        verify(empresaService).deleteEmpresa(1L);
    }

    @Test
    void testDeleteEmpresa_NotFound() {
        doThrow(new RuntimeException("Empresa not found")).when(empresaService).deleteEmpresa(999L);

        assertThrows(RuntimeException.class, () -> {
            empresaController.deleteEmpresa(999L);
        });

        verify(empresaService).deleteEmpresa(999L);
    }

    @Test
    void testDeleteEmpresa_NegativeId() {
        doThrow(new IllegalArgumentException("ID must be positive")).when(empresaService).deleteEmpresa(-1L);

        assertThrows(IllegalArgumentException.class, () -> {
            empresaController.deleteEmpresa(-1L);
        });

        verify(empresaService).deleteEmpresa(-1L);
    }

    @Test
    void testDeleteEmpresa_ServiceException() {
        doThrow(new RuntimeException("Service error")).when(empresaService).deleteEmpresa(1L);

        assertThrows(RuntimeException.class, () -> {
            empresaController.deleteEmpresa(1L);
        });

        verify(empresaService).deleteEmpresa(1L);
    }

    @Test
    void testDeleteEmpresa_ConstraintViolation() {
        doThrow(new RuntimeException("Cannot delete empresa with existing projects"))
            .when(empresaService).deleteEmpresa(1L);

        assertThrows(RuntimeException.class, () -> {
            empresaController.deleteEmpresa(1L);
        });

        verify(empresaService).deleteEmpresa(1L);
    }

    // Testes adicionais para cobertura de edge cases
    @Test
    void testGetAllEmpresas_MultipleEmpresas() {
        Empresa empresa2 = new Empresa();
        empresa2.setId(2L);
        empresa2.setNome("Empresa 2");
        empresa2.setCnpj("98.765.432/0001-10");

        List<Empresa> empresas = Arrays.asList(empresa, empresa2);
        when(empresaService.getAllEmpresas()).thenReturn(empresas);

        List<Empresa> result = empresaController.getAllEmpresas();

        assertNotNull(result);
        assertEquals(2, result.size());
        assertEquals(empresa, result.get(0));
        assertEquals(empresa2, result.get(1));
        verify(empresaService).getAllEmpresas();
    }

    @Test
    void testUpdateEmpresa_PartialUpdate() {
        EmpresaDTO empresaDTOParcial = new EmpresaDTO();
        empresaDTOParcial.setNome("Novo Nome");
        // CNPJ e email permanecem os mesmos

        Empresa empresaAtualizada = new Empresa();
        empresaAtualizada.setId(1L);
        empresaAtualizada.setNome("Novo Nome");
        empresaAtualizada.setCnpj("12.345.678/0001-90");
        empresaAtualizada.setEmail("contato@empresateste.com");

        when(empresaService.updateEmpresa(1L, empresaDTOParcial)).thenReturn(empresaAtualizada);

        Empresa result = empresaController.updateEmpresa(1L, empresaDTOParcial);

        assertNotNull(result);
        assertEquals("Novo Nome", result.getNome());
        assertEquals("12.345.678/0001-90", result.getCnpj());
        assertEquals("contato@empresateste.com", result.getEmail());
        verify(empresaService).updateEmpresa(1L, empresaDTOParcial);
    }
}