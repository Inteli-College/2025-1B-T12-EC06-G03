package com.athenas.athenas.repository;

import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.model.ResponsavelProjeto;
import com.athenas.athenas.model.Usuario;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ResponsavelProjetoRepository extends JpaRepository<ResponsavelProjeto, Long> {
    List<ResponsavelProjeto> findByProjeto(Projeto projeto);
    List<ResponsavelProjeto> findByUsuario(Usuario usuario);
}
