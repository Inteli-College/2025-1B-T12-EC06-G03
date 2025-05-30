package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Entity
@Table(name = "projetos")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Projeto {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String nome;

    private String descricao; // já adicionado anteriormente

    private String status; // ✅ novo campo

    @ManyToOne
    @JoinColumn(name = "empresa_id")
    private Empresa empresa; // ✅ novo campo

    @OneToMany(mappedBy = "projeto", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Imagem> imagens;
}
