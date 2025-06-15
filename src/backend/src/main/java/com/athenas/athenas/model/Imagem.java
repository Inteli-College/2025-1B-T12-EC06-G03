package com.athenas.athenas.model;

import java.time.LocalDateTime;

import org.hibernate.annotations.CreationTimestamp;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "imagens")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Imagem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "fachada_id")
    @JsonIgnoreProperties({"imagens"}) // evita recursividade mas permite acesso à fachada e ao edifício
    private Fachada fachada;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
    @JsonIgnoreProperties({"imagens", "edificios"}) // evita loop
    private Projeto projeto;

    @Column(name = "caminho_arquivo")
    private String caminhoArquivo;

    @Column(name = "nome_arquivo")
    private String nomeArquivo;

    @Column(name = "data_captura")
    @CreationTimestamp
    private LocalDateTime dataCaptura;

    @Column(name = "data_upload")
    @CreationTimestamp
    private LocalDateTime dataUpload;

    @Column(name = "metadados")
    private String metadados;

    @Column(nullable = false)
    private Boolean processada = false;

    @Column(name = "processada_por")
    private String processadaPor;
}
