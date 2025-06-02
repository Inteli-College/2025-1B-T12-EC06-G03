package com.athenas.athenas.controller;

import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.service.AuthService;

import lombok.AllArgsConstructor;
import lombok.Data;

@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private AuthService authService;

    @PostMapping("/login")
    public ResponseEntity<Object> login(@RequestBody LoginRequest request) {
        try {
            Usuario usuario = authService.autenticar(request.getEmail(), request.getSenha());
            String token = authService.gerarToken(usuario.getEmail());
            return ResponseEntity.ok(new LoginResponse(token, usuario));
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(new ErrorResponse(e.getMessage(), "LOGIN_ERROR"));
        }
    }


    @PostMapping("/register")
    public ResponseEntity<Object> register(@RequestBody Usuario usuario) {
        try {
            Usuario novoUsuario = authService.registrar(usuario);
            return ResponseEntity.ok(novoUsuario);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    @GetMapping("/@me")
    public ResponseEntity<Object> me(@RequestHeader("Authorization") String authHeader) {
        Optional<Usuario> usuarioOpt = authService.getUsuarioFromToken(authHeader);

        if (usuarioOpt.isEmpty()) {
            return ResponseEntity.status(401).build();
        }
        return ResponseEntity.ok(usuarioOpt.get());
    }

    @Data
    public static class LoginRequest {
        private String email;
        private String senha;
    }

    @Data
    @AllArgsConstructor
    public static class ErrorResponse {
        private String message;
        private String code;
    }

    @Data
    @AllArgsConstructor
    public static class LoginResponse {
        private String token;
        private Usuario usuario;
    }
}
