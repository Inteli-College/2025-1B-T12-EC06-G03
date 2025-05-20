package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "fachadas")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Fachada {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "edificio_id")
    private Edificio edificio;
    
    private String nome;
    private Double area;
    private String descricao;
}
