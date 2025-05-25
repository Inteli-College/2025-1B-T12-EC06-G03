package com.athenas.athenas.repository;

import com.athenas.athenas.model.LogAlteracao;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.model.Usuario;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface LogAlteracaoRepository extends JpaRepository<LogAlteracao, Long> {
    List<LogAlteracao> findByProjeto(Projeto projeto);
    List<LogAlteracao> findByUsuario(Usuario usuario);
    List<LogAlteracao> findByTipoAlteracao(String tipoAlteracao);
    List<LogAlteracao> findByEntidadeAfetada(String entidadeAfetada);
    List<LogAlteracao> findByDataAlteracaoBetween(LocalDateTime inicio, LocalDateTime fim);
}
