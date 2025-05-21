package com.athenas.athenas.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;

@Repository
public interface EdificioRepository extends JpaRepository<Edificio, Long> {
    Optional<List<Edificio>> findByProjeto(Projeto projeto);
    List<Edificio> findByTipo(String tipo);
}
