package com.athenas.athenas.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.Projeto;

@Repository
public interface ProjetoRepository extends JpaRepository<Projeto, Long> {
    List<Projeto> findByEmpresa(Empresa empresa);
    List<Projeto> findByStatus(String status);
    Optional<Projeto> findById(int idProjeto);
    List<Projeto> findByNome(String nome);
    List<Projeto> findByNomeIgnoreCase(String nome);
    List<Projeto> findByStatusAndEmpresa(String status, Empresa empresa);

    @Query("SELECT p FROM Projeto p WHERE lower(p.nome) = lower(:nome)")
    List<Projeto> buscarPorNomeSemAcento(@Param("nome") String nome);
}
