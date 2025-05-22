package com.athenas.athenas.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.service.ImageService;
import com.athenas.athenas.service.ProjetoService;


@RestController
@RequestMapping("/api/images")
public class ImageController {
    private final ImageService imageService;
    private final ProjetoService projetoService;

    public ImageController(ImageService imageService, ProjetoService projetoService) {
        this.imageService = imageService;
        this.projetoService = projetoService;
    }
    
    @PostMapping(path = "/{projectId}/upload/{edificioId}/{direction}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Void> uploadFiles(@PathVariable Long projectId, @PathVariable String direction, @PathVariable Long edificioId,
            @RequestParam("files") List<MultipartFile> files) {
        if (!projetoService.findById(projectId).isPresent()) {
            return ResponseEntity.notFound().build();
        }
        Projeto projeto = projetoService.findById(projectId).get();
        try {
            for (MultipartFile file : files) {
                imageService.uploadFile(projeto, direction, edificioId, file);
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
        return ResponseEntity.ok().build();
    }

    @GetMapping("/{projectId}")
    public ResponseEntity<List<Imagem>> getImagesByProjectId(@PathVariable Long projectId) {
        if (!projetoService.findById(projectId).isPresent()) {
            return ResponseEntity.notFound().build();
        }
        Projeto projeto = projetoService.findById(projectId).get();
        List<Imagem> images = imageService.getImagesByProject(projeto);
        return ResponseEntity.ok(images);
    }
    
}
