package com.athenas.athenas.model;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

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
    @JsonIgnoreProperties({"edificios"}) // evita loop de serialização
    private Projeto projeto;
    
    private String nome;
    private String localizacao;
    private String tipo;
    private Integer pavimentos;
    
    @OneToMany(mappedBy = "edificio", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    @JsonIgnoreProperties({"edificio"}) // evita loop de serialização
    private List<Fachada> fachadas;
}
