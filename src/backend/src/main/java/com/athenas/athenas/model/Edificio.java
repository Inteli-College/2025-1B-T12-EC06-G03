package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "edificios")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Edificio {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
    private Projeto projeto;
    
    private String nome;
    private String localizacao;
    private String tipo;
    private Integer pavimentos;
}
