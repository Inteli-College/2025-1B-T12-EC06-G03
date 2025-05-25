package com.athenas.athenas.repository;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FachadaRepository extends JpaRepository<Fachada, Long> {
    List<Fachada> findByEdificio(Edificio edificio);
}
