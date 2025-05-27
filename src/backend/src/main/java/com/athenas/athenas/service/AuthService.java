package com.athenas.athenas.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import com.athenas.athenas.model.Usuario;
import com.athenas.athenas.repository.UsuarioRepository;
import com.athenas.athenas.utils.JwtUtil;

@Service
public class AuthService {

    @Autowired
    private UsuarioRepository usuarioRepository;

    @Autowired
    private JwtUtil jwtUtil;

    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public Usuario autenticar(String email, String senha) throws Exception {
        Optional<Usuario> usuarioOpt = usuarioRepository.findByEmail(email);

        if (usuarioOpt.isEmpty()) {
            System.out.println("Usuário não encontrado para email: " + email);
            throw new Exception("Usuário não encontrado");
        }

        Usuario usuario = usuarioOpt.get();

        System.out.println("Hash salva no banco: " + usuario.getSenha());
        System.out.println("Senha enviada para autenticar: " + senha);

        boolean matches = passwordEncoder.matches(senha, usuario.getSenha());
        System.out.println("Senha confere? " + matches);

        if (!matches) {
            throw new Exception("Senha inválida");
        }

        usuario.setUltimoAcesso(LocalDateTime.now());
        usuarioRepository.save(usuario);

        return usuario;
    }

    public String gerarToken(String email) {
        return jwtUtil.generateToken(email);
    }

    public Usuario registrar(Usuario usuario) throws Exception {
        if (usuarioRepository.findByEmail(usuario.getEmail()).isPresent()) {
            throw new Exception("Email já cadastrado");
        }

        System.out.println("Senha antes de codificar: " + usuario.getSenha());
        usuario.setSenha(passwordEncoder.encode(usuario.getSenha()));
        System.out.println("Senha codificada para salvar: " + usuario.getSenha());

        usuario.setDataCriacao(LocalDateTime.now());
        usuario.setUltimoAcesso(LocalDateTime.now());

        return usuarioRepository.save(usuario);
    }

    public Optional<Usuario> getUsuarioFromToken(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return Optional.empty();
        }
        String token = authHeader.substring(7);
        if (!jwtUtil.isTokenValid(token)) {
            return Optional.empty();
        }
        String email = jwtUtil.extractEmail(token);
        return usuarioRepository.findByEmail(email);
    }
}
