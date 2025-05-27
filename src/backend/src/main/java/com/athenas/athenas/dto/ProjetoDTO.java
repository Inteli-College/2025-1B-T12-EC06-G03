package com.athenas.athenas.DTO;

import lombok.*;

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
