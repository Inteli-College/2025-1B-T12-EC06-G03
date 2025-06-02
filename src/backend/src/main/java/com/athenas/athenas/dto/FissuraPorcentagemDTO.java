package com.athenas.athenas.dto;

import java.util.Map;

public class FissuraPorcentagemDTO {
    private Map<String, Integer> porcentagemPorTipo;

    public Map<String, Integer> getPorcentagemPorTipo() { return porcentagemPorTipo; }
    public void setPorcentagemPorTipo(Map<String, Integer> porcentagemPorTipo) { this.porcentagemPorTipo = porcentagemPorTipo; }
}