package com.athenas.athenas.config;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import org.springframework.context.annotation.Configuration;

@Configuration
@OpenAPIDefinition(
    info = @Info(
        title = "Athenas API",
        version = "1.0",
        description = "Documentação automática da API com Swagger OpenAPI 3"
    )
)
public class OpenApiConfig {
    // Configuração básica do OpenAPI/Swagger
}
