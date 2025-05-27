package com.athenas.athenas.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.model.Projeto;
import com.athenas.athenas.repository.EdificioRepository;
import com.athenas.athenas.repository.EmpresaRepository;
import com.athenas.athenas.repository.FachadaRepository;
import com.athenas.athenas.repository.ImagemRepository;
import com.athenas.athenas.repository.ProjetoRepository;

@Service
public class ProjetoService {

    private final ProjetoRepository projetoRepository;
    private final EmpresaRepository empresaRepository;
    private final FachadaRepository fachadaRepository;
    private final ImagemRepository imagemRepository;
    private final EdificioRepository edificioRepository;

    @Value("${supabase.project.url}")
    private String supabaseProjectUrl;

    @Value("${supabase.bucket.name}")
    private String supabaseBucketName;

    @Value("${supabase.service.role.key}")
    private String supabaseServiceRoleKey;

    public ProjetoService(ProjetoRepository projetoRepository, EmpresaRepository empresaRepository,
            EdificioRepository edificioRepository,
            FachadaRepository fachadaRepository, ImagemRepository imagemRepository) {
        this.imagemRepository = imagemRepository;
        this.projetoRepository = projetoRepository;
        this.empresaRepository = empresaRepository;
        this.fachadaRepository = fachadaRepository;
        this.edificioRepository = edificioRepository;
    }

    public List<Projeto> findAll() {
        return projetoRepository.findAll()
                .stream()
                .sorted((p1, p2) -> p1.getId().compareTo(p2.getId()))
                .toList();
    }

    public Optional<Projeto> findById(Long id) {
        return projetoRepository.findById(id);
    }

    public List<Projeto> findByEmpresa(Empresa empresa) {
        return projetoRepository.findByEmpresa(empresa);
    }

    public List<Projeto> findByStatus(String status) {
        return projetoRepository.findByStatus(status);
    }

    public List<Projeto> findByNome(String nome) {
        return projetoRepository.findByNome(nome);
    }

    public List<Projeto> findByStatusAndEmpresa(String status, Empresa empresa) {
        return projetoRepository.findByStatusAndEmpresa(status, empresa);
    }

    public Projeto save(Projeto projeto) {
        return projetoRepository.save(projeto);
    }

    public void delete(Long id) {
        projetoRepository.deleteById(id);
    }

    public Projeto update(Projeto projeto) {
        return projetoRepository.save(projeto);
    }

    public Projeto saveWithEmpresaId(Projeto projeto, Long empresaId) {
        Empresa empresa = empresaRepository.findById(empresaId)
                .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada com id: " + empresaId));
        projeto.setEmpresa(empresa);
        return projetoRepository.save(projeto);
    }

    public Projeto updateWithEmpresaId(Projeto projeto, Long empresaId) {
        Empresa empresa = empresaRepository.findById(empresaId)
                .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada com id: " + empresaId));
        projeto.setEmpresa(empresa);
        return projetoRepository.save(projeto);
    }

}
