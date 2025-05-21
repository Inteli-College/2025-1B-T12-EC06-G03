package com.athenas.athenas.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.Projeto;

@Repository
public interface ProjetoRepository extends JpaRepository<Projeto, Long> {
    List<Projeto> findByEmpresa(Empresa empresa);
    List<Projeto> findByStatus(String status);
    List<Projeto> findByNome(String nome);
    List<Projeto> findByStatusAndEmpresa(String status, Empresa empresa);
}
