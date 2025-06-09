package com.athenas.athenas.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

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
    @JsonIgnoreProperties({"fachada", "metadados", "dataCaptura", "dataUpload", "processada", "imagens"})
    private Imagem imagem;

    private String tipo;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "json")
    private String coordenadas;

    private String gravidade;

    @Column(name = "data_deteccao")
    private LocalDateTime dataDeteccao;

    private Double confianca;

    // Campo opcional para integração com botão de aprovação
    private Boolean aprovado;

    private String aprovadoPor;
}
