package com.athenas.athenas.service;

import java.io.IOException;
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

    public void uploadFile(Projeto projeto, String direction, Long edificioId ,MultipartFile file) {
        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            logger.error("Tipo de arquivo inválido: {}", contentType);
            throw new IllegalArgumentException("Apenas arquivos de imagem são permitidos.");
        }

        String fileName = file.getOriginalFilename();
        String filePath = projeto.getId() + "/" + direction + "/" + fileName;
        String url = String.format("%s/storage/v1/object/%s/%s", supabaseProjectUrl, supabaseBucketName, filePath);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.parseMediaType(contentType));
        headers.setBearerAuth(supabaseServiceRoleKey);
        headers.set("x-upsert", "true");

        try {
            logger.info("Iniciando upload do arquivo {} para o caminho {}", fileName, filePath);
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

            Imagem imagem = new Imagem();
            imagem.setCaminhoArquivo(filePath);
            imagem.setNomeArquivo(fileName);
            imagem.setFachada(fachadaRepository.findByEdificioAndNome(edificio, direction));
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
}
