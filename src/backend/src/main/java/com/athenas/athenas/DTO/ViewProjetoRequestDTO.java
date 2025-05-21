package com.athenas.athenas.DTO;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ViewProjetoRequestDTO {
    @JsonProperty("idProjeto")
    private Integer idProjeto;

    public Integer getIdProjeto() {
        return idProjeto;
    }

    public void setIdProjeto(Integer idProjeto) {
        this.idProjeto = idProjeto;
    }
}