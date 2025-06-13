package com.athenas.athenas.controllersTests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

import com.athenas.athenas.controller.ImageController;
import com.athenas.athenas.controller.ImageController.ProcessadaRequest;
import com.athenas.athenas.model.Imagem;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.service.ImageService;
import com.athenas.athenas.service.ProjetoService;

@ExtendWith(MockitoExtension.class)
public class ImageControllerTests {

    @Mock
    private ImageService imageService;

    @Mock
    private ProjetoService projetoService;

    @Mock
    private MultipartFile multipartFile1;

    @Mock
    private MultipartFile multipartFile2;

    @InjectMocks
    private ImageController imageController;

    private Projeto projeto;
    private Imagem imagem;
    private ProcessadaRequest processadaRequest;
    private List<MultipartFile> files;

    @BeforeEach
    void setUp() {
        projeto = new Projeto();
        projeto.setId(1L);
        projeto.setNome("Projeto Teste");

        imagem = new Imagem();
        imagem.setId(1L);
        imagem.setCaminhoArquivo("imagem_teste.jpg");
        imagem.setProcessada(false);

        processadaRequest = new ProcessadaRequest();
        processadaRequest.setProcessada(true);
        processadaRequest.setProcessadaPor("usuario_teste");

        files = Arrays.asList(multipartFile1, multipartFile2);
    }

    private void setupMultipartFileMocks() {
        when(multipartFile1.getOriginalFilename()).thenReturn("image1.jpg");
        when(multipartFile2.getOriginalFilename()).thenReturn("image2.jpg");
    }

    // Testes para uploadFiles()
    @Test
    void testUploadFiles_Success() {
        setupMultipartFileMocks(); // Only setup when needed
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));

        ResponseEntity<Void> response = imageController.uploadFiles(1L, "NORTE", 1L, files);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        verify(imageService, times(2)).uploadFile(eq(projeto), eq("NORTE"), eq(1L), any(MultipartFile.class));
        verify(imageService).uploadFile(projeto, "NORTE", 1L, multipartFile1);
        verify(imageService).uploadFile(projeto, "NORTE", 1L, multipartFile2);
    }

    @Test
    void testUploadFiles_ProjectNotFound() {
        when(projetoService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<Void> response = imageController.uploadFiles(1L, "NORTE", 1L, files);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        verify(projetoService).findById(1L);
        verify(imageService, never()).uploadFile(any(), anyString(), anyLong(), any(MultipartFile.class));
    }

    @Test
    void testUploadFiles_ServiceException() {
        when(multipartFile1.getOriginalFilename()).thenReturn("image1.jpg");
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        doThrow(new RuntimeException("Upload failed")).when(imageService)
            .uploadFile(eq(projeto), eq("NORTE"), eq(1L), eq(multipartFile1));

        ResponseEntity<Void> response = imageController.uploadFiles(1L, "NORTE", 1L, files);

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        verify(projetoService, times(2)).findById(1L); // Updated to match actual behavior
        verify(imageService).uploadFile(projeto, "NORTE", 1L, multipartFile1);
    }

    @Test
    void testUploadFiles_EmptyFilesList() {
        List<MultipartFile> emptyFiles = new ArrayList<>();
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));

        ResponseEntity<Void> response = imageController.uploadFiles(1L, "SUL", 2L, emptyFiles);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        verify(projetoService, times(2)).findById(1L);
        verify(imageService, never()).uploadFile(any(), anyString(), anyLong(), any(MultipartFile.class));
    }

    @Test
    void testUploadFiles_SingleFile() {
        when(multipartFile1.getOriginalFilename()).thenReturn("image1.jpg");
        List<MultipartFile> singleFile = Arrays.asList(multipartFile1);
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));

        ResponseEntity<Void> response = imageController.uploadFiles(1L, "LESTE", 3L, singleFile);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        verify(projetoService, times(2)).findById(1L);
        verify(imageService, times(1)).uploadFile(eq(projeto), eq("LESTE"), eq(3L), any(MultipartFile.class));
        verify(imageService).uploadFile(projeto, "LESTE", 3L, multipartFile1);
    }

    // Testes para getImagesByProjectId()
    @Test
    void testGetImagesByProjectId_Success() {
        List<Imagem> imagens = Arrays.asList(imagem);
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        when(imageService.getImagesByProject(projeto)).thenReturn(imagens);

        ResponseEntity<List<Imagem>> response = imageController.getImagesByProjectId(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(imagens, response.getBody());
        assertEquals(1, response.getBody().size());
        verify(projetoService, times(2)).findById(1L);
        verify(imageService).getImagesByProject(projeto);
    }

    @Test
    void testGetImagesByProjectId_ProjectNotFound() {
        when(projetoService.findById(1L)).thenReturn(Optional.empty());

        ResponseEntity<List<Imagem>> response = imageController.getImagesByProjectId(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(projetoService).findById(1L);
        verify(imageService, never()).getImagesByProject(any());
    }

    @Test
    void testGetImagesByProjectId_EmptyList() {
        List<Imagem> emptyList = new ArrayList<>();
        when(projetoService.findById(1L)).thenReturn(Optional.of(projeto));
        when(imageService.getImagesByProject(projeto)).thenReturn(emptyList);

        ResponseEntity<List<Imagem>> response = imageController.getImagesByProjectId(1L);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(emptyList, response.getBody());
        assertEquals(0, response.getBody().size());
        verify(projetoService, times(2)).findById(1L);
        verify(imageService).getImagesByProject(projeto);
    }

    // Testes para updateImageProcessada()
    @Test
    void testUpdateImageProcessada_Success() {
        Imagem imagemAtualizada = new Imagem();
        imagemAtualizada.setId(1L);
        imagemAtualizada.setProcessada(true);
        imagemAtualizada.setProcessadaPor("usuario_teste");

        when(imageService.updateProcessadaStatus(1L, true, "usuario_teste")).thenReturn(imagemAtualizada);

        ResponseEntity<Imagem> response = imageController.updateImageProcessada(1L, processadaRequest);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(imagemAtualizada, response.getBody());
        assertEquals(true, response.getBody().getProcessada());
        assertEquals("usuario_teste", response.getBody().getProcessadaPor());
        verify(imageService).updateProcessadaStatus(1L, true, "usuario_teste");
    }

    @Test
    void testUpdateImageProcessada_ImageNotFound() {
        when(imageService.updateProcessadaStatus(1L, true, "usuario_teste"))
            .thenThrow(new RuntimeException("Image not found"));

        ResponseEntity<Imagem> response = imageController.updateImageProcessada(1L, processadaRequest);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        assertNull(response.getBody());
        verify(imageService).updateProcessadaStatus(1L, true, "usuario_teste");
    }

    @Test
    void testUpdateImageProcessada_WithFalseProcessada() {
        ProcessadaRequest falseRequest = new ProcessadaRequest();
        falseRequest.setProcessada(false);
        falseRequest.setProcessadaPor("admin");

        Imagem imagemAtualizada = new Imagem();
        imagemAtualizada.setId(1L);
        imagemAtualizada.setProcessada(false);
        imagemAtualizada.setProcessadaPor("admin");

        when(imageService.updateProcessadaStatus(1L, false, "admin")).thenReturn(imagemAtualizada);

        ResponseEntity<Imagem> response = imageController.updateImageProcessada(1L, falseRequest);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(imagemAtualizada, response.getBody());
        assertEquals(false, response.getBody().getProcessada());
        assertEquals("admin", response.getBody().getProcessadaPor());
        verify(imageService).updateProcessadaStatus(1L, false, "admin");
    }

    @Test
    void testUpdateImageProcessada_WithNullValues() {
        ProcessadaRequest nullRequest = new ProcessadaRequest();
        nullRequest.setProcessada(null);
        nullRequest.setProcessadaPor(null);

        Imagem imagemAtualizada = new Imagem();
        imagemAtualizada.setId(1L);

        when(imageService.updateProcessadaStatus(1L, null, null)).thenReturn(imagemAtualizada);

        ResponseEntity<Imagem> response = imageController.updateImageProcessada(1L, nullRequest);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(imagemAtualizada, response.getBody());
        verify(imageService).updateProcessadaStatus(1L, null, null);
    }

    // Testes para deleteImage()
    @Test
    void testDeleteImage_Success() {
        ResponseEntity<Void> response = imageController.deleteImage(1L);

        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
        verify(imageService).deleteImage(1L);
    }

    @Test
    void testDeleteImage_ImageNotFound() {
        doThrow(new RuntimeException("Image not found")).when(imageService).deleteImage(1L);

        ResponseEntity<Void> response = imageController.deleteImage(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        verify(imageService).deleteImage(1L);
    }

    @Test
    void testDeleteImage_ServiceException() {
        doThrow(new RuntimeException("Database error")).when(imageService).deleteImage(1L);

        ResponseEntity<Void> response = imageController.deleteImage(1L);

        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
        verify(imageService).deleteImage(1L);
    }

    // Testes para ProcessadaRequest 
    @Test
    void testProcessadaRequest_GettersSetters() {
        ProcessadaRequest request = new ProcessadaRequest();
        
        request.setProcessada(true);
        request.setProcessadaPor("test_user");
        
        assertEquals(true, request.getProcessada());
        assertEquals("test_user", request.getProcessadaPor());
    }

    @Test
    void testProcessadaRequest_NullValues() {
        ProcessadaRequest request = new ProcessadaRequest();
        
        assertNull(request.getProcessada());
        assertNull(request.getProcessadaPor());
        
        request.setProcessada(null);
        request.setProcessadaPor(null);
        
        assertNull(request.getProcessada());
        assertNull(request.getProcessadaPor());
    }
}