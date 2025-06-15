package com.athenas.athenas.service;

import java.io.IOException;
import java.text.Normalizer;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import com.athenas.athenas.model.Edificio;
import com.athenas.athenas.model.Fachada;
import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.EmpresaRepository;
import com.athenas.athenas.repository.FachadaRepository;
import com.athenas.athenas.repository.ImagemRepository;
import com.athenas.athenas.repository.ProjetoRepository;

@Service
public class ImageService {
    private static final Logger logger = LoggerFactory.getLogger(ImageService.class);

    private final FachadaRepository fachadaRepository;
    private final ImagemRepository imagemRepository;
    private final EdificioRepository edificioRepository;

    @Value("${supabase.project.url}")
    private String supabaseProjectUrl;

    @Value("${supabase.bucket.name}")
    private String supabaseBucketName;

    @Value("${supabase.service.role.key}")
    private String supabaseServiceRoleKey;

    public ImageService(ProjetoRepository projetoRepository, EmpresaRepository empresaRepository,
                        EdificioRepository edificioRepository,
                        FachadaRepository fachadaRepository, ImagemRepository imagemRepository) {
        this.imagemRepository = imagemRepository;
        this.fachadaRepository = fachadaRepository;
        this.edificioRepository = edificioRepository;
    }

    public List<Imagem> getImagesByProject(Projeto projeto) {
        return imagemRepository.findByProjeto(projeto);
    }

    /**
     * Sanitiza uma string para ser usada como nome de arquivo/pasta
     * Remove acentos, espaços e caracteres especiais
     */
    private String sanitizeFileName(String input) {
        if (input == null) {
            return "sem-nome";
        }
        
        // Remove acentos
        String normalized = Normalizer.normalize(input, Normalizer.Form.NFD);
        String withoutAccents = normalized.replaceAll("\\p{InCombiningDiacriticalMarks}+", "");
        
        // Remove espaços e caracteres especiais, mantém apenas letras, números e hífens
        String sanitized = withoutAccents.replaceAll("[^a-zA-Z0-9-]", "-");
        
        // Remove hífens múltiplos e hífens no início/fim
        sanitized = sanitized.replaceAll("-+", "-").replaceAll("^-|-$", "");
        
        // Se ficar vazio, retorna um nome padrão
        if (sanitized.isEmpty()) {
            return "sem-nome";
        }
        
        return sanitized.toLowerCase();
    }

    public void uploadFile(Projeto projeto, String direction, Long edificioId, MultipartFile file) {
        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            logger.error("Tipo de arquivo inválido: {}", contentType);
            throw new IllegalArgumentException("Apenas arquivos de imagem são permitidos.");
        }

        String fileName = file.getOriginalFilename();
        if (fileName == null) {
            fileName = "imagem-sem-nome.jpg";
        }
        
        // Sanitiza o nome do arquivo
        String sanitizedFileName = sanitizeFileName(fileName);
        
        // Sanitiza a direção (descrição da fachada)
        String sanitizedDirection = sanitizeFileName(direction);
        
        String filePath = projeto.getId() + "/" + sanitizedDirection + "/" + sanitizedFileName;
        String url = String.format("%s/storage/v1/object/%s/%s", supabaseProjectUrl, supabaseBucketName, filePath);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.parseMediaType(contentType));
        headers.setBearerAuth(supabaseServiceRoleKey);
        headers.set("x-upsert", "true");

        try {
            logger.info("Iniciando upload do arquivo {} para o caminho sanitizado {}", fileName, filePath);
            HttpEntity<byte[]> entity = new HttpEntity<>(file.getBytes(), headers);
            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.PUT, entity, String.class);

            if (!response.getStatusCode().is2xxSuccessful()) {
                logger.error("Erro ao fazer upload para Supabase. Status: {}, Body: {}", response.getStatusCode(), response.getBody());
                throw new RuntimeException("Erro ao fazer upload para Supabase: " + response.getStatusCode());
            }

            Edificio edificio = edificioRepository.findById(edificioId)
                .orElseThrow(() -> {
                    logger.error("Edificio não encontrado com id: {}", edificioId);
                    return new RuntimeException("Edificio não encontrado com id: " + edificioId);
                });

            // Busca a fachada pela descrição original (não sanitizada)
            Fachada fachada = fachadaRepository.findByEdificioAndNome(edificio, direction);
            if (fachada == null) {
                // Se não encontrar pela descrição exata, busca todas as fachadas do edifício
                List<Fachada> fachadasDoEdificio = fachadaRepository.findByEdificio(edificio);
                if (!fachadasDoEdificio.isEmpty()) {
                    // Usa a primeira fachada encontrada como fallback
                    fachada = fachadasDoEdificio.get(0);
                    logger.warn("Fachada com nome '{}' não encontrada, usando fachada '{}' como fallback", 
                              direction, fachada.getDescricao());
                } else {
                    logger.error("Nenhuma fachada encontrada para o edifício {}", edificioId);
                    throw new RuntimeException("Nenhuma fachada encontrada para o edifício " + edificioId);
                }
            }

            Imagem imagem = new Imagem();
            imagem.setCaminhoArquivo(filePath);
            imagem.setNomeArquivo(fileName); // Mantém o nome original
            imagem.setFachada(fachada);
            imagem.setProjeto(projeto);
            imagemRepository.save(imagem);
            logger.info("Upload e persistência da imagem realizados com sucesso para o arquivo {}", fileName);
        } catch (IOException e) {
            logger.error("Falha ao ler o arquivo para upload: {}", e.getMessage(), e);
            throw new RuntimeException("Falha ao ler o arquivo para upload", e);
        } catch (org.springframework.web.client.HttpClientErrorException | 
                 org.springframework.web.client.HttpServerErrorException e) {
            logger.error("Erro HTTP ao fazer upload para Supabase: {}", e.getMessage(), e);
            throw new RuntimeException("Erro HTTP ao fazer upload para Supabase: " + e.getMessage(), e);
        } catch (java.util.NoSuchElementException e) {
            logger.error("Erro ao acessar elemento inexistente: {}", e.getMessage(), e);
            throw new RuntimeException("Erro ao acessar elemento inexistente: " + e.getMessage(), e);
        } catch (RuntimeException e) {
            logger.error("Falha inesperada no upload do arquivo para Supabase: {}", e.getMessage(), e);
            throw new RuntimeException("Falha inesperada no upload do arquivo para Supabase", e);
        }
    }

    public void deleteImageById(Long id) {
        imagemRepository.deleteById(id);
    }

    public Imagem updateProcessadaStatus(Long imageId, Boolean processada, String processadaPor) {
        Imagem imagem = imagemRepository.findById(imageId)
            .orElseThrow(() -> new RuntimeException("Imagem não encontrada com id: " + imageId));
        
        imagem.setProcessada(processada);
        imagem.setProcessadaPor(processadaPor);
        
        return imagemRepository.save(imagem);
    }

    public void deleteImage(Long imageId) {
        Imagem imagem = imagemRepository.findById(imageId)
            .orElseThrow(() -> new RuntimeException("Imagem não encontrada com id: " + imageId));
        
        // Delete the file from Supabase storage
        String filePath = imagem.getCaminhoArquivo();
        if (filePath != null && !filePath.isEmpty()) {
            deleteFromSupabase(filePath);
        }
        
        imagemRepository.delete(imagem);
    }

    /**
     * Deleta um arquivo do Supabase Storage
     */
    private void deleteFromSupabase(String filePath) {
        try {
            // Validação das configurações
            if (supabaseProjectUrl == null || supabaseProjectUrl.isEmpty()) {
                logger.error("supabaseProjectUrl não configurado");
                return;
            }
            if (supabaseBucketName == null || supabaseBucketName.isEmpty()) {
                logger.error("supabaseBucketName não configurado");
                return;
            }
            if (supabaseServiceRoleKey == null || supabaseServiceRoleKey.isEmpty()) {
                logger.error("supabaseServiceRoleKey não configurado");
                return;
            }

            String url = String.format("%s/storage/v1/object/%s/%s", supabaseProjectUrl, supabaseBucketName, filePath);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(supabaseServiceRoleKey);
            
            HttpEntity<String> entity = new HttpEntity<>(headers);
            RestTemplate restTemplate = new RestTemplate();
            
            logger.info("Deletando arquivo do Supabase: {} | URL: {}", filePath, url);
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, entity, String.class);
            
            if (response.getStatusCode().is2xxSuccessful()) {
                logger.info("Arquivo deletado com sucesso do Supabase: {}", filePath);
            } else if (response.getStatusCode().value() == 404) {
                logger.warn("Arquivo não encontrado no Supabase (pode já ter sido deletado): {}", filePath);
            } else {
                logger.warn("Falha ao deletar arquivo do Supabase. Status: {}, Response: {}", 
                           response.getStatusCode(), response.getBody());
            }
        } catch (org.springframework.web.client.HttpClientErrorException.NotFound e) {
            logger.warn("Arquivo não encontrado no Supabase (404): {}", filePath);
        } catch (org.springframework.web.client.HttpClientErrorException | 
                 org.springframework.web.client.HttpServerErrorException e) {
            logger.error("Erro HTTP ao deletar arquivo do Supabase: {} - Status: {}, Response: {}", 
                        filePath, e.getStatusCode(), e.getResponseBodyAsString());
        } catch (Exception e) {
            logger.error("Erro inesperado ao deletar arquivo do Supabase: {}", filePath, e);
        }
    }
}
