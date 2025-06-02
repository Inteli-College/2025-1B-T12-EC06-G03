package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "fissuras")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Fissura {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "imagem_id")
    private Imagem imagem;
    
    private String tipo;
    
    @Column(columnDefinition = "json")
    private String coordenadas;
    
    private String gravidade;
    
    @Column(name = "data_deteccao")
    private LocalDateTime dataDeteccao;
    
    private Double confianca;
}
