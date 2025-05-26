package com.athenas.athenas.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.athenas.athenas.dto.EmpresaDTO;
import com.athenas.athenas.model.Empresa;
import com.athenas.athenas.repository.EmpresaRepository;

@Service
public class EmpresaService {
    
    @Autowired
    private EmpresaRepository empresaRepository;
    
    public Empresa createEmpresa(EmpresaDTO empresaDTO) {
        Empresa empresa = new Empresa();
        empresa.setNome(empresaDTO.getNome());
        empresa.setCnpj(empresaDTO.getCnpj());
        empresa.setEndereco(empresaDTO.getEndereco());
        empresa.setTelefone(empresaDTO.getTelefone());
        empresa.setEmail(empresaDTO.getEmail());
        return empresaRepository.save(empresa);
    }
    
    public Empresa getEmpresaById(int idEmpresa) {
        return empresaRepository.findById(idEmpresa)
            .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada com id: " + idEmpresa));
    }
    
    public Empresa getEmpresaByNome(String nomeEmpresa) {
        return empresaRepository.findByNome(nomeEmpresa)
            .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada com nome: " + nomeEmpresa));
    }
    
    public Empresa getEmpresaByCNPJ(String cnpjEmpresa) {
        return empresaRepository.findByCnpj(cnpjEmpresa)
            .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada com cnpj: " + cnpjEmpresa));
    }
    
    public List<Empresa> getAllEmpresas() {
        return empresaRepository.findAll();
    }

    public Empresa updateEmpresa(long idEmpresa, EmpresaDTO empresaDTO) {
        Empresa empresa = empresaRepository.findById(idEmpresa)
            .orElseThrow(() -> new IllegalArgumentException("Empresa não encontrada para atualização"));
    
        if (empresaDTO.getNome() != null) {
            empresa.setNome(empresaDTO.getNome());
        }
        if (empresaDTO.getCnpj() != null) {
            empresa.setCnpj(empresaDTO.getCnpj());
        }
        if (empresaDTO.getEndereco() != null) {
            empresa.setEndereco(empresaDTO.getEndereco());
        }
        if (empresaDTO.getTelefone() != null) {
            empresa.setTelefone(empresaDTO.getTelefone());
        }
        if (empresaDTO.getEmail() != null) {
            empresa.setEmail(empresaDTO.getEmail());
        }
        return empresaRepository.save(empresa);
    }
    
    public void deleteEmpresa(Long idEmpresa) {
        if (empresaRepository.existsById(idEmpresa)) {
            empresaRepository.deleteById(idEmpresa);
        } else {
            throw new IllegalArgumentException("Empresa não encontrada com id: " + idEmpresa);
        }
    }
}