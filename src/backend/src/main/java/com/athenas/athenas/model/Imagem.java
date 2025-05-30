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

    @ManyToOne
    @JoinColumn(name = "fachada_id")
    @JsonIgnoreProperties({"imagens", "edificio"}) // evita recursividade com fachada
    private Fachada fachada;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
    @JsonIgnoreProperties({"imagens"}) // evita loop e permite acesso a projeto.nome
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
}
