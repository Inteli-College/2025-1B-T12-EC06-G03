package com.athenas.athenas.repository;

import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.Projeto;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ProjetoRepository extends JpaRepository<Projeto, Long> {
    List<Projeto> findByEmpresa(Empresa empresa);
    List<Projeto> findByStatus(String status);
}
