package com.athenas.athenas.controller;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.multipart.MultipartFile;

import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.service.ImageService;
import com.athenas.athenas.service.ProjetoService;

@RestController
@RequestMapping("/api/images")
public class ImageController {
    private static final Logger logger = LoggerFactory.getLogger(ImageController.class);

    private final ImageService imageService;
    private final ProjetoService projetoService;

    public ImageController(ImageService imageService, ProjetoService projetoService) {
        this.imageService = imageService;
        this.projetoService = projetoService;
    }

    @PostMapping(path = "/{projectId}/upload/{edificioId}/{direction}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> uploadFiles(@PathVariable Long projectId, @PathVariable("direction") String direction, @PathVariable("edificioId") Long edificioId,
            @RequestParam("files") List<MultipartFile> files) {
        logger.info("Received upload request: projectId={}, edificioId={}, direction={}, filesCount={}", projectId, edificioId, direction, files.size());
        if (!projetoService.findById(projectId).isPresent()) {
            logger.warn("Project with id {} not found", projectId);
            return ResponseEntity.notFound().build();
        }
        Projeto projeto = projetoService.findById(projectId).get();
        try {
            for (MultipartFile file : files) {
                logger.info("Uploading file: name={}, size={}", file.getOriginalFilename(), file.getSize());
                imageService.uploadFile(projeto, direction, edificioId, file);
            }
        } catch (Exception e) {
            logger.error("Error uploading files for projectId {}: {}", projectId, e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
        logger.info("Successfully uploaded {} files for projectId {}", files.size(), projectId);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{projectId}")
    public ResponseEntity<List<Imagem>> getImagesByProjectId(@PathVariable Long projectId) {
        logger.info("Fetching images for projectId {}", projectId);
        if (!projetoService.findById(projectId).isPresent()) {
            logger.warn("Project with id {} not found", projectId);
            return ResponseEntity.notFound().build();
        }
        Projeto projeto = projetoService.findById(projectId).get();
        List<Imagem> images = imageService.getImagesByProject(projeto);
        logger.info("Found {} images for projectId {}", images.size(), projectId);
        return ResponseEntity.ok(images);
    }
}
