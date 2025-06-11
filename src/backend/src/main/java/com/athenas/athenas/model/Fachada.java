package com.athenas.athenas.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

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
    @JsonIgnoreProperties({"fachadas"}) // evita loop de serialização
    private Edificio edificio;
    
    private String nome;
    private Double area;
    private String descricao;
}
