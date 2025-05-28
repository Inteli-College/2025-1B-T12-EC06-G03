package com.athenas.athenas.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter

public class ProjetoDTO {
    private String nome;
    private String descricao;
    private String status;
    private Long empresa;
}
