package com.athenas.athenas.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Projeto;

@Repository
public interface EdificioRepository extends JpaRepository<Edificio, Long> {
    
    List<Edificio> findByProjeto(Projeto projeto);

    List<Edificio> findByProjetoId(Long projetoId);
    
    List<Edificio> findByTipo(String tipo);

    List<Edificio> findByProjetoAndTipo(Projeto projeto, String tipo);

    List<Edificio> findByProjetoIdAndTipo(Long projetoId, String tipo);
    
    Optional<Edificio> findByNome(String nome);

    List<Edificio> findByNomeContainingIgnoreCase(String nome);

    @Modifying
    @Transactional
    void deleteByProjeto(Projeto projeto);

    @Modifying
    @Transactional
    void deleteByProjetoId(Long projetoId);
    
    @Modifying
    @Transactional
    void deleteByTipo(String tipo);
    
    @Modifying
    @Transactional
    void deleteByProjetoAndTipo(Projeto projeto, String tipo);
    
    @Modifying
    @Transactional
    void deleteByNome(String nome);    

}