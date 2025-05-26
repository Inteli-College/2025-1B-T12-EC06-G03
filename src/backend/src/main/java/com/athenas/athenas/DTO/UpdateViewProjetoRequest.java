package com.athenas.athenas.DTO;

public class UpdateViewProjetoRequest {
    private int idProjeto;
    private ViewProjetoResponseDTO viewProjetoResponseDTO;

    public void setIdProjeto(int idProjeto){
        this.idProjeto = idProjeto;
    }

    public int getIdProjeto(){
        return this.idProjeto;
    }

    public void setViewProjetoResponseDTO(ViewProjetoResponseDTO viewProjetoResponseDTO){
        this.viewProjetoResponseDTO = viewProjetoResponseDTO;
    }

    public ViewProjetoResponseDTO getViewProjetoResponseDTO(){
        return this.viewProjetoResponseDTO;
    }
}
