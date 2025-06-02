package com.athenas.athenas.dto;

import java.util.Map;

public class FissuraDetalheDTO {
    private Long id;
    private String tipo;
    private String coordenadas;
    private String gravidade;
    private String dataDeteccao;
    private Double confianca;
    private String nomeImagem;
    private Map<String, Integer> porcentagemPorTipo;
    private Boolean processada;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTipo() { return tipo; }
    public void setTipo(String tipo) { this.tipo = tipo; }
    public String getCoordenadas() { return coordenadas; }
    public void setCoordenadas(String coordenadas) { this.coordenadas = coordenadas; }
    public String getGravidade() { return gravidade; }
    public void setGravidade(String gravidade) { this.gravidade = gravidade; }
    public String getDataDeteccao() { return dataDeteccao; }
    public void setDataDeteccao(String dataDeteccao) { this.dataDeteccao = dataDeteccao; }
    public Double getConfianca() { return confianca; }
    public void setConfianca(Double confianca) { this.confianca = confianca; }
    public String getNomeImagem() { return nomeImagem; }
    public void setNomeImagem(String nomeImagem) { this.nomeImagem = nomeImagem; }
    public Map<String, Integer> getPorcentagemPorTipo() { return porcentagemPorTipo; }
    public void setPorcentagemPorTipo(Map<String, Integer> porcentagemPorTipo) { this.porcentagemPorTipo = porcentagemPorTipo; }
    public Boolean getProcessada() { return processada; }
    public void setProcessada(Boolean processada) { this.processada = processada; }
}