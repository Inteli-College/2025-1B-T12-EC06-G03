package com.athenas.athenas.repository;

import com.athenas.athenas.model.Fissura;
import com.athenas.athenas.model.Imagem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FissuraRepository extends JpaRepository<Fissura, Long> {
    List<Fissura> findByImagem(Imagem imagem);
    List<Fissura> findByTipo(String tipo);
    List<Fissura> findByGravidade(String gravidade);
    List<Fissura> findByConfiancaGreaterThan(Double confianca);
}
