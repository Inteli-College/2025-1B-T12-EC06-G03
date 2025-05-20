package com.athenas.athenas.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

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
    private Fachada fachada;
    
    @Column(name = "caminho_arquivo")
    private String caminhoArquivo;
    
    @Column(name = "nome_arquivo")
    private String nomeArquivo;
    
    @Column(name = "data_captura")
    private LocalDateTime dataCaptura;
    
    @Column(name = "data_upload")
    private LocalDateTime dataUpload;
    
    @Column(columnDefinition = "json")
    private String metadados;
    
    private Boolean processada;
}
