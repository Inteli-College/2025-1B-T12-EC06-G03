package com.athenas.athenas.model;

import java.time.LocalDateTime;

import org.hibernate.annotations.CreationTimestamp;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
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
    private Fachada fachada;

    @ManyToOne
    @JoinColumn(name = "projeto_id")
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
