package com.athenas.athenas.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.repository.UsuarioRepository;

@Service
public class AuthService {

    @Autowired
    private UsuarioRepository usuarioRepository;

    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public Usuario autenticar(String email, String senha) throws Exception {
        Optional<Usuario> usuarioOpt = usuarioRepository.findByEmail(email);

        if (usuarioOpt.isEmpty()) {
            throw new Exception("Usuário não encontrado");
        }

        Usuario usuario = usuarioOpt.get();

        if (!passwordEncoder.matches(senha, usuario.getSenha())) {
            throw new Exception("Senha inválida");
        }

        usuario.setUltimoAcesso(LocalDateTime.now());
        usuarioRepository.save(usuario);

        return usuario;
    }

    public Usuario registrar(Usuario usuario) throws Exception {
        if (usuarioRepository.findByEmail(usuario.getEmail()).isPresent()) {
            throw new Exception("Email já cadastrado");
        }

        usuario.setSenha(passwordEncoder.encode(usuario.getSenha()));
        usuario.setDataCriacao(LocalDateTime.now());
        usuario.setUltimoAcesso(LocalDateTime.now());

        return usuarioRepository.save(usuario);
    }
}
