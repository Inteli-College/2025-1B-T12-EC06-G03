package com.athenas.athenas.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.athenas.athenas.DTO.UpdateViewProjetoRequest;
import com.athenas.athenas.DTO.ViewProjetoRequestDTO;
import com.athenas.athenas.DTO.ViewProjetoResponseDTO;
import com.athenas.athenas.service.ViewProjetoService;

@RestController
@RequestMapping("/api/projeto")
public class ViewProjetoController {

    private final ViewProjetoService viewProjetoService;

    @Autowired
    public ViewProjetoController(ViewProjetoService viewProjetoService) {
        this.viewProjetoService = viewProjetoService;
    }

    @PostMapping("/ViewProjeto")
    public ResponseEntity<ViewProjetoResponseDTO> viewProjeto(@RequestBody ViewProjetoRequestDTO viewProjetoRequestDTO) {
        try {
            ViewProjetoResponseDTO response = this.viewProjetoService.ReadProjectData(viewProjetoRequestDTO.getIdProjeto());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PutMapping("/UpdateViewProjeto")
    public ViewProjetoResponseDTO updateViewProjeto (@RequestBody UpdateViewProjetoRequest updateViewProjetoRequest){
        return viewProjetoService.UpdateProjectData(updateViewProjetoRequest.getIdProjeto(), updateViewProjetoRequest.getViewProjetoResponseDTO());
    }
}
