package com.athenas.athenas.service;

import java.io.IOException;
import java.util.List;

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
            HttpEntity<byte[]> entity = new HttpEntity<>(file.getBytes(), headers);
            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.PUT, entity, String.class);

            if (!response.getStatusCode().is2xxSuccessful()) {
                throw new RuntimeException("Erro ao fazer upload para Supabase: " + response.getStatusCode());
            }

            Edificio edificio = edificioRepository.findById(edificioId)
                .orElseThrow(() -> new RuntimeException("Edificio não encontrado com id: " + edificioId));

            Imagem imagem = new Imagem();
            imagem.setCaminhoArquivo(filePath);
            imagem.setNomeArquivo(fileName);
            imagem.setFachada(fachadaRepository.findByEdificioAndNome(edificio, direction));
            imagem.setProjeto(projeto);
            imagemRepository.save(imagem);
        } catch (IOException e) {
            throw new RuntimeException("Falha ao ler o arquivo para upload", e);
        } catch (RuntimeException e) {
            throw new RuntimeException("Falha no upload do arquivo para Supabase", e);
        }
    }
}
