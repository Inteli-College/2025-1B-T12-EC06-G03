package com.athenas.athenas.repository;

import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.model.Imagem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface ImagemRepository extends JpaRepository<Imagem, Long> {
    List<Imagem> findByFachada(Fachada fachada);
    List<Imagem> findByProcessada(Boolean processada);
    List<Imagem> findByDataCapturaBetween(LocalDateTime inicio, LocalDateTime fim);
}
