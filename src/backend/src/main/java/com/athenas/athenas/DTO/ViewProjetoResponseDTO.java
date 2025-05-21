package com.athenas.athenas.DTO;

import java.util.List;

import com.athenas.athenas.model.Edificio;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ViewProjetoResponseDTO {
    @JsonProperty("projeto")
    private String nome;
    
    @JsonProperty("responsaveis")
    private List<String> responsaveisNomes;
    
    @JsonProperty("empresa")
    private String empresaNome;
    
    @JsonProperty("edificios")
    private List<Edificio> edificios;
    
    @JsonProperty("descricao")
    private String descricao;
    
    @JsonProperty("logs_alteracoes")
    private List<String> logs;

    public String getNome() {
        return nome;
    }

    public void setNome(String nome) {
        this.nome = nome;
    }

    public List<String> getResponsaveisNomes() {
        return responsaveisNomes;
    }

    public void setResponsaveisNomes(List<String> responsaveisNomes) {
        this.responsaveisNomes = responsaveisNomes;
    }

    public String getEmpresaNome() {
        return empresaNome;
    }

    public void setEmpresaNome(String empresaNome) {
        this.empresaNome = empresaNome;
    }

    public List<Edificio> getEdificios() {
        return edificios;
    }

    public void setEdificios(List<Edificio> edificios) {
        this.edificios = edificios;
    }

    public String getDescricao() {
        return descricao;
    }

    public void setDescricao(String descricao) {
        this.descricao = descricao;
    }

    public List<String> getLogs() {
        return logs;
    }

    public void setLogs(List<String> logs) {
        this.logs = logs;
    }
}