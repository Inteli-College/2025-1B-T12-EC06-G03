package com.athenas.athenas.repository;

import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.model.Relatorio;
import com.athenas.athenas.model.Usuario;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface RelatorioRepository extends JpaRepository<Relatorio, Long> {
    List<Relatorio> findByProjeto(Projeto projeto);
    List<Relatorio> findByUsuario(Usuario usuario);
    List<Relatorio> findByDataGeracaoBetween(LocalDateTime inicio, LocalDateTime fim);
}
