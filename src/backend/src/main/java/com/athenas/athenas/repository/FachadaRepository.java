package com.athenas.athenas.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;

@Repository
public interface FachadaRepository extends JpaRepository<Fachada, Long> {
    List<Fachada> findByEdificio(Edificio edificio);
    Fachada findByEdificioAndNome(Edificio edificio, String nome);
}
