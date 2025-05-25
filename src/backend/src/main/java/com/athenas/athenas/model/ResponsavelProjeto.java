package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "responsaveis_projeto")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ResponsavelProjeto {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
    private Projeto projeto;
    
    @ManyToOne
    @JoinColumn(name = "usuario_id")
    private Usuario usuario;
}
