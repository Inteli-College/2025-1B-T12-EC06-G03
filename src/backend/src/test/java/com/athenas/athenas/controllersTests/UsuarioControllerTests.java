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
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import com.athenas.athenas.controller.UsuarioController;
import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.service.UsuarioService;

@ExtendWith(MockitoExtension.class)
public class UsuarioControllerTests {

    @Mock
    private UsuarioService usuarioService;

    @InjectMocks
    private UsuarioController usuarioController;

    private Usuario usuario;

    @BeforeEach
    void setUp() {
        usuario = new Usuario();
        usuario.setId(1L);
        usuario.setNome("Usuario Teste");
        usuario.setEmail("usuario@teste.com");
    }

    // Testes para getAllUsuarios()
    @Test
    void testGetAllUsuarios_Success() {
        List<Usuario> usuarios = Arrays.asList(usuario);
        when(usuarioService.findAll()).thenReturn(usuarios);

        List<Usuario> response = usuarioController.getAllUsuarios();

        assertEquals(usuarios, response);
        verify(usuarioService).findAll();
    }

    @Test
    void testGetAllUsuarios_EmptyList() {
        List<Usuario> usuariosVazios = new ArrayList<>();
        when(usuarioService.findAll()).thenReturn(usuariosVazios);

        List<Usuario> response = usuarioController.getAllUsuarios();

        assertEquals(usuariosVazios, response);
        verify(usuarioService).findAll();
    }

    @Test
    void testGetAllUsuarios_ServiceException() {
        when(usuarioService.findAll()).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.getAllUsuarios();
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).findAll();
    }

    // Testes para getUsuarioById()
    @Test
    void testGetUsuarioById_Success() {
        when(usuarioService.findById(1L)).thenReturn(Optional.of(usuario));

        ResponseEntity<Usuario> response = usuarioController.getUsuarioById(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(usuario, response.getBody());
        verify(usuarioService).findById(1L);
    }

    @Test
    void testGetUsuarioById_NotFound() {
        when(usuarioService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Usuario> response = usuarioController.getUsuarioById(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).findById(1L);
    }

    @Test
    void testGetUsuarioById_ServiceException() {
        when(usuarioService.findById(1L)).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.getUsuarioById(1L);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).findById(1L);
    }

    // Testes para getUsuarioByEmail()
    @Test
    void testGetUsuarioByEmail_Success() {
        when(usuarioService.findByEmail("usuario@teste.com")).thenReturn(Optional.of(usuario));

        ResponseEntity<Usuario> response = usuarioController.getUsuarioByEmail("usuario@teste.com");

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(usuario, response.getBody());
        verify(usuarioService).findByEmail("usuario@teste.com");
    }

    @Test
    void testGetUsuarioByEmail_NotFound() {
        when(usuarioService.findByEmail("inexistente@teste.com")).thenReturn(Optional.empty());

        ResponseEntity<Usuario> response = usuarioController.getUsuarioByEmail("inexistente@teste.com");

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).findByEmail("inexistente@teste.com");
    }

    @Test
    void testGetUsuarioByEmail_ServiceException() {
        when(usuarioService.findByEmail("usuario@teste.com")).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.getUsuarioByEmail("usuario@teste.com");
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).findByEmail("usuario@teste.com");
    }

    // Testes para createUsuario()
    @Test
    void testCreateUsuario_Success() {
        when(usuarioService.save(any(Usuario.class))).thenReturn(usuario);

        Usuario response = usuarioController.createUsuario(usuario);

        assertEquals(usuario, response);
        verify(usuarioService).save(any(Usuario.class));
    }

    @Test
    void testCreateUsuario_WithNullValues() {
        Usuario usuarioComNulos = new Usuario();
        usuarioComNulos.setId(null);
        usuarioComNulos.setNome(null);
        usuarioComNulos.setEmail(null);

        when(usuarioService.save(any(Usuario.class))).thenReturn(usuarioComNulos);

        Usuario response = usuarioController.createUsuario(usuarioComNulos);

        assertEquals(usuarioComNulos, response);
        verify(usuarioService).save(any(Usuario.class));
    }

    @Test
    void testCreateUsuario_ServiceException() {
        when(usuarioService.save(any(Usuario.class))).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.createUsuario(usuario);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).save(any(Usuario.class));
    }

    // Testes para updateUsuario()
    @Test
    void testUpdateUsuario_Success() {
        Usuario usuarioAtualizado = new Usuario();
        usuarioAtualizado.setId(1L);
        usuarioAtualizado.setNome("Usuario Atualizado");
        usuarioAtualizado.setEmail("atualizado@teste.com");

        when(usuarioService.findById(1L)).thenReturn(Optional.of(usuario));
        when(usuarioService.save(any(Usuario.class))).thenReturn(usuarioAtualizado);

        ResponseEntity<Usuario> response = usuarioController.updateUsuario(1L, usuarioAtualizado);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(usuarioAtualizado, response.getBody());
        assertEquals(1L, usuarioAtualizado.getId()); // Verifica se o ID foi setado
        verify(usuarioService).findById(1L);
        verify(usuarioService).save(any(Usuario.class));
    }

    @Test
    void testUpdateUsuario_NotFound() {
        when(usuarioService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Usuario> response = usuarioController.updateUsuario(1L, usuario);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).findById(1L);
        verify(usuarioService, never()).save(any(Usuario.class));
    }

    @Test
    void testUpdateUsuario_ServiceException() {
        when(usuarioService.findById(1L)).thenReturn(Optional.of(usuario));
        when(usuarioService.save(any(Usuario.class))).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.updateUsuario(1L, usuario);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).findById(1L);
        verify(usuarioService).save(any(Usuario.class));
    }

    // Testes para deleteUsuario()
    @Test
    void testDeleteUsuario_Success() {
        when(usuarioService.findById(1L)).thenReturn(Optional.of(usuario));

        ResponseEntity<Void> response = usuarioController.deleteUsuario(1L);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).findById(1L);
        verify(usuarioService).delete(1L);
    }

    @Test
    void testDeleteUsuario_NotFound() {
        when(usuarioService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Void> response = usuarioController.deleteUsuario(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).findById(1L);
        verify(usuarioService, never()).delete(anyLong());
    }

    @Test
    void testDeleteUsuario_ServiceException() {
        when(usuarioService.findById(1L)).thenReturn(Optional.of(usuario));
        doThrow(new RuntimeException("Service error")).when(usuarioService).delete(1L);

        try {
            usuarioController.deleteUsuario(1L);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).findById(1L);
        verify(usuarioService).delete(1L);
    }

    // Testes para updateUltimoAcesso()
    @Test
    void testUpdateUltimoAcesso_Success() {
        Usuario usuarioComUltimoAcesso = new Usuario();
        usuarioComUltimoAcesso.setId(1L);
        usuarioComUltimoAcesso.setNome("Usuario Teste");
        usuarioComUltimoAcesso.setEmail("usuario@teste.com");

        when(usuarioService.atualizarUltimoAcesso(1L)).thenReturn(usuarioComUltimoAcesso);

        ResponseEntity<Usuario> response = usuarioController.updateUltimoAcesso(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(usuarioComUltimoAcesso, response.getBody());
        verify(usuarioService).atualizarUltimoAcesso(1L);
    }

    @Test
    void testUpdateUltimoAcesso_NotFound() {
        when(usuarioService.atualizarUltimoAcesso(1L)).thenReturn(null);

        ResponseEntity<Usuario> response = usuarioController.updateUltimoAcesso(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(usuarioService).atualizarUltimoAcesso(1L);
    }

    @Test
    void testUpdateUltimoAcesso_ServiceException() {
        when(usuarioService.atualizarUltimoAcesso(1L)).thenThrow(new RuntimeException("Service error"));

        try {
            usuarioController.updateUltimoAcesso(1L);
        } catch (RuntimeException e) {
            assertEquals("Service error", e.getMessage());
        }
        
        verify(usuarioService).atualizarUltimoAcesso(1L);
    }
}