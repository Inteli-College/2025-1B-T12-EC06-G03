package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "logs_alteracoes")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LogAlteracao {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
    private Projeto projeto;
    
    @ManyToOne
    @JoinColumn(name = "usuario_id")
    private Usuario usuario;
    
    @Column(name = "tipo_alteracao")
    private String tipoAlteracao;
    
    private String descricao;
    
    @Column(name = "data_alteracao")
    private LocalDateTime dataAlteracao;
    
    @Column(name = "entidade_afetada")
    private String entidadeAfetada;
    
    @Column(name = "entidade_id")
    private Long entidadeId;
}
