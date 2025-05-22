package com.athenas.athenas.DTO;

public class EdificioDTO {
    private String nome;
    private String localizacao;
    private String tipo;
    private int pavimentos;

    public String getNome() {
        return nome;
    }

    public void setNome(String nome) {
        this.nome = nome;
    }

    public String getLocalizacao() {
        return localizacao;
    }

    public void setLocalizacao(String localizacao) {
        this.localizacao = localizacao;
    }

    public String getTipo() {
        return tipo;
    }

    public void setTipo(String tipo) {
        this.tipo = tipo;
    }

    public int getPavimentos() {
        return pavimentos;
    }

    public void setPavimentos(int pavimentos) {
        this.pavimentos = pavimentos;
    }

}