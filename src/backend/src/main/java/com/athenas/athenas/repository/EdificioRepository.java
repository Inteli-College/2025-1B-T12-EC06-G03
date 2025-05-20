package com.athenas.athenas.repository;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface EdificioRepository extends JpaRepository<Edificio, Long> {
    List<Edificio> findByProjeto(Projeto projeto);
    List<Edificio> findByTipo(String tipo);
}
